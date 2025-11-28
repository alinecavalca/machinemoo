# -*- coding: utf-8 -*-
"""
Many Objective Noninferior Estimation (MONISE)

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Raimundo, Marcos M.
    MONISE - Many Objective Noninferior Estimation
    2017
    arXiv
"""

import copy
import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, cast
from numbers import Real

try:
    import gurobipy as gp
    import mip
    # Parameters to silence Gurobi
    params = {
        "LogToConsole": 0,
    }
    try:
        env = gp.Env(empty=True, params=params)
        env.start()
    except gp.GurobiError as e:
        print(f"Gurobi Error: {e}")
except ImportError:
    import mip

from machinemoo.core.base_algorithm import BaseMOO
from machinemoo.utils.typing import scalar
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "Monise"
]

# Large integer for MILP constraints
MAXINT = 2000000000

class WeightNode:
    """
    Solves a scalarization weight optimization problem for multi-objective learning.
    
    This class uses Mixed-Integer Linear Programming (MILP) to estimate a new 
    weighting vector that maximizes the improvement in the Pareto front approximation.
    """
    def __init__(
        self,
        solutions: List[scalar],
        global_lower: npt.NDArray[np.float64],
        global_upper: npt.NDArray[np.float64],
        weighted_scalar: scalar,
        time_limit: float = 10.0,
        mip_gap: float = 0.01, 
    ) -> None:
        """
        Args:
            solutions (List[scalar]): Current Pareto front approximation.
            global_lower (np.ndarray): Utopia point (lower bounds).
            global_upper (np.ndarray): Nadir point (upper bounds).
            weighted_scalar (scalar): Scalarization prototype.
            time_limit (float): Solver time limit.
            mip_gap (float): Solver optimality gap.
        """
        self.weighted_scalar = weighted_scalar
        self.solutions = solutions
        self.M = solutions[0].M
        self.global_lower = global_lower
        self.global_upper = global_upper
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        
        self.best_solution_reached = False
        self.importance: float = 0.0
        self.w: Optional[npt.NDArray[np.float64]] = None
        self._solution: Optional[scalar] = None

        # Determine which calculation strategy to use
        # If we only have the initial individual minima (M solutions), use simpler LP
        if len(self.solutions) == self.M:
            self._calc_first_w()
        else:
            self._calc_w()

    def optimize(self) -> scalar:
        """
        Optimizes using the calculated weight vector.
        Uses the best existing solution (based on the new weight) as a warm start.
        """
        if self.w is None:
            raise RuntimeError("Weight vector 'w' is not set; cannot optimize.")

        best_solution = None
        best_objective = np.inf

        # Find warm start
        for solution in self.solutions:
            # Calculate weighted sum using current w
            val = float(self.w @ solution.objs)
            if val < best_objective:
                best_objective = val
                best_solution = solution

        self._solution = copy.copy(self.weighted_scalar)
        self._solution.optimize(self.w)
        
        # Convergence check: if the "new" solution is identical to the warm start
        if best_solution is not None and np.all(np.isclose(self._solution.objs, best_solution.objs)):
            self.best_solution_reached = True
            
        return self._solution

    def _calc_first_w(self) -> None:
        """
        Solve linear optimization for the initial weight vector.
        Used when only individual minima are present.
        """
        oidx = [i for i in range(self.M)]
        
        # Setup MIP model
        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except Exception:
            prob = mip.Model(sense=mip.MAXIMIZE)
        prob.verbose = 0
        # Variables
        w = [prob.add_var(name='w', var_type=mip.CONTINUOUS, lb=cast(Real, 0.0), ub=cast(Real, 1.0)) for _ in oidx]
        v = prob.add_var(name='v', var_type=mip.CONTINUOUS)
        v = prob.add_var(name='v', var_type=mip.CONTINUOUS)
        
        # Utopia point (normalized)
        # uR is effectively 0 vector in normalized space if global_lower is accurate
        # But we calculate it explicitly to be safe
        uR_norm = self.global_lower

        for sols in self.solutions:
            # Use objs_lb for robust lower bound estimation logic
            # If not available, fallback to objs via property
            objs_norm = sols.objs_lb
            
            expr = v - mip.xsum(w[i] * objs_norm[i] for i in oidx)
            prob += expr <= cast(Real, 0.0)

        # Sum of weights = 1
        prob += mip.xsum(w[i] for i in oidx) == 1
        
        # Distance to Utopia
        # v - w @ uR
        prob += v - mip.xsum(w[i] * uR_norm[i] for i in oidx)

        status = prob.optimize()
        feasible = status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]

        if feasible:
            w_val = np.array([var.x if var.x >= cast(Real, 0.0) else 0 for var in w])
                
            self.w = w_val / w_val.sum()
                
            obj_val = prob.objective_value
            self.importance = float(obj_val) if obj_val is not None else 0.0
        else:
            # Fallback
            self.w = np.ones(self.M) / self.M
            self.importance = 0.0

    def _calc_w(self, eps: float = 0.00) -> None:
        """
        Solve MILP to find the next optimal weight vector.
        """
        oidx = [i for i in range(self.M)]
        n_sols = len(self.solutions)

        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except Exception:
            prob = mip.Model(sense=mip.MAXIMIZE)
        prob.verbose = 0

        # Variables
        w = [prob.add_var(name='w', lb=cast(Real, 0.0), var_type=mip.CONTINUOUS) for _ in oidx]
        uR = [prob.add_var(name='uR', lb=cast(Real, -np.inf), var_type=mip.CONTINUOUS) for _ in oidx]
        kp = [prob.add_var(name='kp', lb=cast(Real, 0.0), var_type=mip.CONTINUOUS) for _ in range(n_sols)]
        nu = [prob.add_var(name='nu', lb=cast(Real, 0.0), var_type=mip.CONTINUOUS) for _ in oidx]
        
        kpB = [prob.add_var(name='kpB', var_type=mip.BINARY) for _ in range(n_sols)]
        nuB = [prob.add_var(name='nuB', var_type=mip.BINARY) for _ in oidx]

        v = prob.add_var(name='v', lb=cast(Real, 0.0), var_type=mip.CONTINUOUS)
        mu = prob.add_var(name='mu', lb=cast(Real, -np.inf), var_type=mip.CONTINUOUS)

        # Pre-calculate normalized values to keep loop clean
        norm_w_list = []
        for idx, s in enumerate(self.solutions):
            # Try to obtain a weight vector from the solution; if unavailable or ill-typed, fall back.
            w_val = getattr(s, "w", None)
            if w_val is None:
                # If this is one of the individual minima, prefer a unit vector for that objective index
                if idx < self.M:
                    w_val = np.zeros(self.M, dtype=float)
                    w_val[idx] = 1.0
                else:
                    # Default to uniform weights
                    w_val = np.ones(self.M, dtype=float) / float(self.M)
            else:
                # Ensure it's a float ndarray of the correct shape
                try:
                    w_val = np.asarray(w_val, dtype=float)
                    if w_val.ndim == 0:
                        w_val = np.ones(self.M, dtype=float) / float(self.M)
                    elif w_val.size != self.M:
                        w_val = w_val.reshape(self.M)
                except Exception:
                    w_val = np.ones(self.M, dtype=float) / float(self.M)
            norm_w_list.append(w_val)
        
        # Use objs_lb (Lipschitz) if available for tighter bounds
        norm_objs_list = [
            s.objs_lb
            for s in self.solutions
        ]
        
        # 1. Constraints on uR (Hyperplane approximations)
        for idx_sol, sols in enumerate(self.solutions):
            # w_sol @ uR >= w_sol @ objs_sol
            expr = mip.xsum(norm_w_list[idx_sol][i] * uR[i] for i in oidx)
            cons = norm_w_list[idx_sol] @ norm_objs_list[idx_sol]
            prob += expr >= cons * (1 - eps)

        # 2. Definition of uR based on active constraints (KKT-like logic)
        for i in oidx:
            expr = uR[i] - mip.xsum(kp[k] * norm_objs_list[k][i] for k in range(n_sols)) - nu[i] + mu
            prob += expr == 0

        # 3. uR must be >= global lower bound (Utopia)
        norm_global_l = self.global_lower
        for i in oidx:
            prob += uR[i] >= norm_global_l[i]

        # Big-M constant
        norm_global_u = self.global_upper
        big_c = np.max(norm_global_u) if len(norm_global_u) > 0 else 100.0

        # 4. Selection constraints (only one region active)
        for k in range(n_sols):
            # w @ objs_k - v >= 0
            expr = mip.xsum(w[i] * norm_objs_list[k][i] for i in oidx) - v
            prob += expr >= cast(Real, 0.0)
            prob += expr <= kpB[k] * cast(Real, big_c)
            prob += kp[k] >= cast(Real, 0.0)
            # avoid subtracting a Var from a scalar (some solvers/types don't support it)
            # enforce kp[k] + kpB[k] <= 1 instead of kp[k] <= 1 - kpB[k]
            prob += kp[k] + kpB[k] <= cast(Real, 1)

        # Simplex constraint
        prob += mip.xsum(w[i] for i in oidx) == 1
        prob += mip.xsum(kp[k] for k in range(n_sols)) == 1

        # 5. Dual constraints
        for i in oidx:
            prob += w[i] >= cast(Real, 0.0)
            prob += w[i] <= nuB[i]
            prob += nu[i] >= cast(Real, 0.0)
            # Avoid direct subtraction between literal and Var (some solvers/types don't support it).
            # Rewrite nu[i] <= (1 - nuB[i]) * 2 * big_c as nu[i] + 2*big_c*nuB[i] <= 2*big_c
            prob += mip.xsum([nu[i], cast(Real, 2 * big_c) * nuB[i]]) <= cast(Real, 2 * big_c)
        
        prob += mip.xsum([mu])
        prob += mu <= v

        # Warm start
        rnd = np.array(sorted([0] + [np.random.rand() for _ in range(self.M - 1)] + [1]))
        w_ini = np.array([rnd[i+1] - rnd[i] for i in range(self.M)])
        w_ini = w_ini / w_ini.sum()
        prob.start = [(w[i], w_ini[i]) for i in range(self.M)]

        # Solver config
        prob.threads = 1
        prob.max_solutions = MAXINT
        prob.max_seconds = self.time_limit

        status = prob.optimize()
        feasible = status in [mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE]
        if feasible:
            w_res = np.array([var.x if var.x >= cast(Real, 0.0) else 0 for var in w])                
            if w_res.sum() > 0:
                self.w = w_res / w_res.sum()
            else:
                self.w = np.ones(self.M) / self.M
                
            obj_val = prob.objective_value
            self.importance = float(obj_val) if obj_val is not None else 0.0
        else:
            self.w = np.ones(self.M) / self.M
            self.importance = 0.0
            self.importance = 0.0


class Monise(BaseMOO):
    """
    MONISE: Many-Objective Non-Inferior Set Estimation.
    
    Incrementally constructs a Pareto frontier approximation by solving a
    sequence of weighted scalarization problems using MILP to find the next
    most promising weight vector.
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar,
        target_size: int = 20,
        time_limit: float = float('inf'),   
        node_time_limit: float = float('inf'),
        node_gap: float = 0.01,
        norm: bool = True,
        hotstart: List[scalar] = [],
        verbose: bool = False,
        debug: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Args:
            weighted_scalar: Strategy for weighted sum.
            single_scalar: Strategy for single objective.
            target_size: Desired number of solutions.
            time_limit: Overall time limit.
            node_time_limit: Time limit per MILP solve.
            node_gap: MIP gap for MILP solve.
            norm: Normalize objectives.
            hotstart: Initial solutions.
        """
        # Initialize BaseMOO
        super().__init__(time_limit=time_limit, target_size=target_size, verbose=verbose, debug=debug)

        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, single_interface))):
            raise ValueError("Scalarizers must implement correct interfaces.")

        self.weighted_scalar = weighted_scalar
        self.single_scalar = single_scalar
        self.node_time_limit = node_time_limit
        self.node_gap = node_gap
        self.hotstart = hotstart
        
        # State
        self.M: int = 0
        self.global_lower: Optional[npt.NDArray[np.float64]] = None
        self.global_upper: Optional[npt.NDArray[np.float64]] = None
        self.importances: List[float] = []
        
        # Used to hold the next node to process
        self._next_node: Optional[WeightNode] = None

    def initialize(self) -> None:
        """
        Finds individual minima, computes bounds, and creates the first WeightNode.
        """
        self.M = self.single_scalar.M
        neig_o = []
        
        # 1. Find Individual Minima
        for i in range(self.M):
            single_s = copy.copy(self.single_scalar)
            self.logger.debug(f"Finding {i+1}th individual minima")
            
            # Attempt hotstart if compatible (requires impl in scalarizer)
            # Standard scalarize doesn't take hotstart in optimize() signature usually
            single_s.optimize(i) 
            
            neig_o.append(single_s.objs)
            # Add to BaseMOO lists (filtering logic applies)
            self.update(None, single_s)

        # 2. Compute Global Bounds
        # Use objs_lb from history if available for tighter Utopia
        objs_lb_matrix = np.array([
            s.objs_lb for s in self.history_list
        ])
        
        # Global Lower (Utopia): min of lower bounds
        self.global_lower = objs_lb_matrix.min(axis=0)
        
        # Global Upper (Nadir): max of actual objectives (safer for normalization)
        objs_matrix = np.array([s.objs for s in self.history_list])
        self.global_upper = objs_matrix.max(axis=0)

        # 3. Create First Node
        # Corresponds to finding the weight that maximizes distance to Utopia
        # given the initial Extreme points.
        first_node = WeightNode(
            solutions=self.solutions_list,
            global_lower=cast(npt.NDArray[np.float64], self.global_lower),
            global_upper=cast(npt.NDArray[np.float64], self.global_upper),
            weighted_scalar=self.weighted_scalar,
        )
        
        self.importances = [first_node.importance]
        
        # Queue this node to be picked up by select()
        self._next_node = first_node

    def select(self) -> Optional[WeightNode]:
        """
        Returns the next WeightNode to optimize.
        Calculates the subsequent node for the *next* iteration before returning.
        """
        # 1. Get the node prepared in previous step
        current_node = self._next_node
        
        # 2. Prepare the *next* node for the future
        # (This implements the iterative nature of Monise loop)
        if current_node is not None:
            # We construct the NEXT search direction based on current state
            # This is slightly different from RandomWeights which is stateless.
            # MONISE logic: Calculate weight for *next* iteration based on current solutions.
            
            new_node = WeightNode(
                solutions=self.solutions_list,
                global_lower=cast(npt.NDArray[np.float64], self.global_lower),
                global_upper=cast(npt.NDArray[np.float64], self.global_upper),
                weighted_scalar=self.weighted_scalar,
                time_limit=self.node_time_limit,
                mip_gap=self.node_gap,
            )
            self.importances.append(new_node.importance)
            self._next_node = new_node

        return current_node

    def update(self, node: Any, solution: scalar) -> None:
        """
        Updates solutions and global lower bound.
        """
        # 1. BaseMOO update (history + filtering)
        super().update(node, solution)
        
        # 2. Update Global Lower Bound (Lipschitz support)
        # If the new solution has a lower estimated bound, expand the Utopia point
        if self.global_lower is not None:
            self.global_lower = np.minimum(self.global_lower, solution.objs_lb)