# -*- coding: utf-8 -*-
"""
Multi-Objective Learning Algorithm (MOLA)

This module implements MOLA, an algorithm that approximates the Pareto front
by iteratively learning scalarization weights. It uses a Mixed-Integer Quadratic
Programming (MIQP) formulation to find weights that maximize the separation
between the upper (achieved) and lower (estimated) Pareto approximations.
"""

import copy
import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional

import pyomo.environ as pyo
from pyomo.contrib import appsi

from machinemoo import get_logger
from machinemoo.moo.core import MOOptimizer, IPSolvableMixin
from machinemoo.utils.typing import scalar
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface

__all__ = [
    "MOLA"
]

class WeightNode:
    """
    Solves the weight optimization problem for MOLA.

    This class encapsulates the MIQP logic to find the optimal weight vector 'w'
    that maximizes the expected improvement in the Pareto front coverage.
    """
    def __init__(
        self,
        solutions: List[scalar],
        global_lower: Optional[npt.NDArray[np.float64]],
        weighted_scalar: scalar,
        time_limit: float = 10.0,
        mip_gap: float = 0.01,
        epsilon: float = 1e-8
    ) -> None:
        """
        Args:
            solutions (List[scalar]): Current Pareto front approximation.
            global_lower (Optional[np.ndarray]): Utopia point (lower bounds). If None, it will be computed
                as the component-wise minimum of the provided solutions' objs_lb.
            weighted_scalar (scalar): Scalarization prototype.
            time_limit (float): Solver time limit.
            mip_gap (float): Solver optimality gap.
            epsilon (float): Small constant to ensure strict dominance constraints.
        """
        self.solutions = solutions
        self.M = solutions[0].M
        # If no global_lower provided, derive a conservative utopia point from solutions' lower bounds.
        if global_lower is None:
            objs_lower_matrix = np.array([s.objs_lb for s in solutions])
            self.global_lower = objs_lower_matrix.min(axis=0)
        else:
            self.global_lower = global_lower
        self.weighted_scalar = weighted_scalar
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.epsilon = epsilon
        
        self.logger = get_logger(f"moo.{__name__}")
        
        self.w: npt.NDArray[np.float64] = np.zeros(self.M)
        self.importance: float = 0.0
        self.best_solution_reached = False
        self._solution: Optional[scalar] = None
        
        self.ml_model: Any = None # Legacy support access

        # Solve for weights immediately upon initialization
        self._compute_weights()

    def optimize(self) -> scalar:
        """
        Optimizes the current solution using the computed weights.
        """
        best_solution = None
        best_objective = np.inf

        # Warm start selection
        for solution in self.solutions:
            val = self.w @ solution.objs
            if val < best_objective:
                best_objective = val
                best_solution = solution

        self._solution = copy.copy(self.weighted_scalar)
        self._solution.optimize(self.w)
        
        self.ml_model = self._solution.x
        
        # Convergence check
        if best_solution is not None and np.all(np.isclose(self._solution.objs, best_solution.objs)):
           self.best_solution_reached = True
           
        return self._solution

    def _compute_weights(self) -> None:
        """
        Solves the MIQP to find the weight vector maximizing the separation margin.
        
        Note: Uses Pyomo + Gurobi because the problem involves quadratic terms 
        (w * y) in the objective/constraints which basic linear solvers don't handle.
        """
        model = pyo.ConcreteModel()
        
        # Data preparation
        objs_list = [s.objs for s in self.solutions]
        # Use lower bounds (Lipschitz) if available
        objs_list_lower = [s.objs_lb for s in self.solutions]
        
        num_objs = self.M
        y_star = self.global_lower

        # Variables
        # b[i, j]: Binary variable for region selection
        model.b = pyo.Var(range(len(objs_list)), range(num_objs), domain=pyo.Binary) # type: ignore
        # w[j]: Scalarization weights
        model.w = pyo.Var(range(num_objs), domain=pyo.NonNegativeReals) # type: ignore
        # y_sup: Upper approximation (achieved frontier)
        model.y_sup = pyo.Var(range(num_objs), domain=pyo.Reals) # type: ignore
        # y_und: Lower approximation (estimated potential)
        model.y_und = pyo.Var(range(num_objs), domain=pyo.Reals) # type: ignore

        model.constraints = pyo.ConstraintList() # type: ignore

        # 1. Upper Bound Constraints (Convex Hull of current solutions)
        # sum(w * (y_sup - y_star)) <= sum(w * (obj - y_star)) for all solutions
        for objs in objs_list:
            lhs = sum(model.w[j] * (model.y_sup[j] - y_star[j]) for j in range(num_objs)) # type: ignore
            rhs = sum(model.w[j] * max(objs[j] - y_star[j], self.epsilon) for j in range(num_objs)) # type: ignore
            model.constraints.add(lhs <= rhs) # type: ignore

        # 2. Lower Bound Constraints (Piecewise definitions via binary vars)
        for i in range(len(objs_list_lower)):
            for j in range(num_objs):
                # if y_und < objs_lower, force b=0
                # Logic: y_und[j] - y_star[j] >= b[i,j] * (objs_lower[i][j] - y_star[j])
                model.constraints.add( # type: ignore 
                    (model.y_und[j] - y_star[j]) >= # type: ignore 
                    model.b[i, j] * (objs_list_lower[i][j] - y_star[j]) # type: ignore 
                )
            
            # Ensure y_sup dominates at least one objs_lower fully?
            # Or ensuring valid partitioning.
            # Original logic: sum(model.b[i, j]) >= 1 
            # implies at least one dimension 'j' respects the bound for solution 'i'
            model.constraints.add( # type: ignore 
                sum(model.b[i, j] for j in range(num_objs)) >= 1 # type: ignore 
            )

        # 3. Consistency Constraints
        for j in range(num_objs):
            # y_sup >= y_und
            model.constraints.add(model.y_sup[j] >= model.y_und[j]) # type: ignore 
            # y_und >= y_star (Utopia)
            model.constraints.add(model.y_und[j] >= y_star[j]) # type: ignore 

        # 4. Simplex Constraint
        model.constraints.add(sum(model.w[j] for j in range(num_objs)) == 1) # type: ignore 

        # Objective: Maximize gap (w * y_sup - w * y_und)
        model.obj = pyo.Objective( # type: ignore 
            expr=sum(model.w[j] * model.y_sup[j] for j in range(num_objs)) - # type: ignore 
                 sum(model.w[j] * model.y_und[j] for j in range(num_objs)), # type: ignore 
            sense=pyo.maximize
        )

        # Solve
        # Using appsi.solvers.Gurobi for persistent interface
        try:
            solver = appsi.solvers.Gurobi()
            solver.config.mip_gap = self.mip_gap
            solver.config.time_limit = self.time_limit
            # Silence Gurobi output if possible via solver options
            # solver.gurobi_options['OutputFlag'] = 0 
            solver.solve(model) # type: ignore 
        except Exception as e:
            self.logger.warning(f"Solver failed: {e}")
            # Fallback to uniform weights
            self.w = np.ones(self.M) / self.M
            return

        # Extract results
        if hasattr(model, 'w'):
            self.w = np.array([pyo.value(model.w[i]) for i in range(num_objs)]) # type: ignore 
            self.importance = pyo.value(model.obj.expr) # type: ignore 
        else:
            self.w = np.ones(self.M) / self.M
            self.importance = 0.0

        # Debug logging
        if self.logger.isEnabledFor(10): # DEBUG
            y_sup_ = np.array([pyo.value(model.y_sup[i]) for i in range(num_objs)]) # type: ignore 
            y_und_ = np.array([pyo.value(model.y_und[i]) for i in range(num_objs)]) # type: ignore 
            self.logger.debug(f"Calculated weights: {self.w}")
            self.logger.debug(f"Importance: {self.importance}")
            self.logger.debug(f"y_sup: {y_sup_}")
            self.logger.debug(f"y_und: {y_und_}")


class MOLA(IPSolvableMixin, MOOptimizer):
    """
    MOLA: Multi-Objective Learning Algorithm.
    
    Inherits from BaseMOO to perform the iterative optimization loop.
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar | None,
        # Base Parameters (MOOPTimizer))
        target_size: int = 20,
        time_limit: float = float('inf'),  
        verbose: bool = False,
        debug: bool = False,
        # IPSolvableMixin Parameters
        node_time_limit: float = float("inf"),
        node_gap: float = 0.01,
    ) -> None:
        """
        Args:
            weighted_scalar: Strategy for weighted sum.
            single_scalar: Strategy for single objective.
            target_size: Desired number of solutions.
            target_gap: Stop if importance gap is small.
            red_fact: Reduction factor for goal (unused in current impl but kept for API).
            smooth_count: Iterations to smooth importance tracking.
            node_time_limit: Time limit per MILP solve.
            node_gap: MIP gap for MILP solve.
            norm: Normalize objectives (unused in MOLA currently, kept for API).
            utopia_slack: Small buffer for bounds.
        """
        super().__init__(
            target_size=target_size,
            time_limit=time_limit,
            verbose=verbose,
            debug=debug,
            node_time_limit=node_time_limit,
            node_gap=node_gap,
        )

        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, w_interface))):
            raise ValueError("Scalarizers must implement correct interfaces.")

        # Setting Scalarizers
        self.weighted_scalar = weighted_scalar
        self.single_scalar = single_scalar
        
        
        # State
        self.M: int = 0
        self.global_lower: Optional[npt.NDArray[np.float64]] = None
        self.importances: List[float] = []
        self._next_node: Optional[WeightNode] = None

    def _update_global_lower(self) -> None:
        """
        Updates the global lower bound (Utopia point) based on the estimated
        lower bounds of all found solutions.
        """
        if not self.history_list:
            return
            
        # Collect 'objs_lower' from all history
        # Note: BaseMOO stores all solutions in self.history_list
        objs_lower_matrix = np.array([s.objs_lb for s in self.history_list])
        
        # Utopia is the component-wise minimum
        self.global_lower = objs_lower_matrix.min(axis=0)

    def initialize(self) -> None:
        """
        Initializes by finding individual minima and creating the first WeightNode.
        """
        self.M = self.single_scalar.M
        
        # 1. Find Individual Minima
        for i in range(self.M):
            self.logger.debug(f"Finding {i+1}th individual minimum")
            single_s = copy.copy(self.single_scalar)
            single_s.optimize(i)
            
            # Add to BaseMOO lists (filtering logic applies)
            self.update(None, single_s)

        # 2. Update Bounds
        self._update_global_lower()

        # 3. Create First Node
        # MOLA logic starts by creating a node considering all current solutions
        first_node = WeightNode(
            solutions=self.solutions_list,
            global_lower=self.global_lower,
            weighted_scalar=self.weighted_scalar,
            time_limit=self.node_time_limit,
            mip_gap=self.node_gap
        )

        self.max_imp = first_node.importance
        self.importances = [first_node.importance]
        
        self._next_node = first_node

    def select(self) -> Optional[WeightNode]:
        """
        Returns the next WeightNode to optimize.
        """
        current_node = self._next_node
        
        if current_node is not None:
            # Prepare the NEXT node for the subsequent iteration
            # This follows the MOLA pattern of recalculating weights based on the NEW set
            # The 'current_node' returned here was calculated in the PREVIOUS step.
            
            # We need to peek ahead: after 'current_node' is optimized in the main loop,
            # 'update' will be called. Then 'select' is called again.
            # So actually, we should calculate the NEW node *here*, inside select, 
            # based on the *current* solutions list (which includes the result of the previous step).
            
            # However, the first node was pre-calculated in initialize().
            # So strictly speaking, select() just pops the queue or generates new one.
            pass

        return current_node

    def optimize_step(self, node: WeightNode) -> scalar:
        """
        Executes optimization and immediately prepares the next node for the loop.
        """
        # 1. Run the optimization for the current node
        solution = node.optimize()
        
        return solution

    def update(self, node: Any, solution: scalar) -> None:
        """
        Updates solutions, bounds, and prepares the next WeightNode.
        """
        # 1. BaseMOO update
        super().update(node, solution)
        
        # 2. Update Bounds
        self._update_global_lower()
        
        # 3. Generate Next Node
        # MOLA generates a new weight vector based on the UPDATED solution set
        new_node = WeightNode(
            solutions=self.solutions_list,
            global_lower=self.global_lower,
            weighted_scalar=self.weighted_scalar,
            time_limit=self.node_time_limit,
            mip_gap=self.node_gap
        )
        
        self.importances.append(new_node.importance)
        self._next_node = new_node