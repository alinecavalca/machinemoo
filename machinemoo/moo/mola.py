import copy
import time

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo
from pyomo.contrib import appsi
from pymoo.indicators.hv import HV

from typing import Any

from machinemoo import get_logger
from machinemoo import scalar_interface, w_interface
from machinemoo.utils.typing import scalar

__all__ = [
    "Mola"
]

logger = get_logger(f"moo.{__name__}")

# For reproducibility, uncomment the line below to fix the NumPy random seed in this file.
np.random.seed(42) # TODO: Comment after dissertation

#EPS = 1e-8


class WeightSolver:
    """Solves a scalarization weight optimization problem for multi-objective learning.

    WeightSolver finds a weight vector that maximizes a separation margin between
    the upper and lower approximations of the Pareto front, using a MILP solver
    (e.g., Gurobi). The computed weights are then used to guide optimization
    in the multi-objective learning algorithm (MOLA).
    """
    def __init__(
        self,
        solutions: list[scalar],
        global_lower: npt.NDArray[np.float64],
        #scalarizer: scalar,
        time_limit: float = 10.0,
        mip_gap: float = 0.01,
        epsilon: float = 1e-8
    ) -> None:
        """Initializes the WeightSolver.

        Args:
            solutions (list[scalar]): Array of candidate solutions with objective vectors.
            global_lower (np.ndarray): Global lower bounds of the objectives.
            scalarizer (scalar): Scalarization function used for optimization.
            time_limit (float): Time limit (in seconds) for solving the MILP problem. Default is 10.0.
            mip_gap (float): Acceptable optimality gap for MILP solver. Default is 0.01.
            epsilon (float): Tolerance for minimum improvement in dominance conditions. Default is 0.05.
        """
        #self._scalarizer = scalarizer
        self._num_objectives = solutions[0].M
        self._global_lower = global_lower
        self._solutions = solutions
        self._time_limit = time_limit
        self._mip_gap =  mip_gap
        self._epsilon = epsilon

        self.ml_model = None
        self.best_solution_reached = False
        self._compute_weights()

    @property
    def M(self) -> int:
        """Number of objective functions.

        Returns:
            int: The number of objectives.
        """
        return self._num_objectives

    @property
    def importance(self) -> float:
        """Importance score computed from the separation margin optimization.

        Returns:
            float: The separation margin between upper and lower bounds.
        """
        return self._importance

    @property
    def parents(self) -> list[scalar]:
        """Candidate solutions used to compute the weights.

        Returns:
            list[scalar]: Array of solution objects used as input.
        """
        return self._solutions

    @property
    def solution(self) -> scalar:
        """The current optimized solution.

        Returns:
            scalar: The best solution found using the computed weights.
        """
        return self._solution

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """The weight vector obtained from the MILP optimization.

        Returns:
            np.ndarray: The vector of weights for scalarization.
        """
        return self._weights

    def get_model(self) -> Any:
        """Returns the underlying machine learning model of the best solution.

        Returns:
            Any: Trained ML model corresponding to the best solution.
        """
        return self.ml_model

    def _compute_weights(self) -> None:
        """Solves the MILP to compute an optimal weight vector for scalarization.

        The optimization maximizes the difference between a weighted upper bound
        and a weighted lower bound over a set of non-dominated solutions. 
        The result is used to guide subsequent solution refinement.
        """
        model = pyo.ConcreteModel()
        objs_list = [s.objs for s in self._solutions]
        objs_list_lower = [s.objs_lower for s in self._solutions]
        num_objs = self._num_objectives

        y_star = self._global_lower

        model.b = pyo.Var(range(len(objs_list)), range(num_objs), domain=pyo.Binary)
        model.w = pyo.Var(range(num_objs), domain=pyo.NonNegativeReals)
        model.y_sup = pyo.Var(range(num_objs), domain=pyo.Reals)
        model.y_und = pyo.Var(range(num_objs), domain=pyo.Reals)

        model.constraints = pyo.ConstraintList()

        for objs in objs_list:
            constr = sum(model.w[j] * (model.y_sup[j] - y_star[j]) for j in range(num_objs)) <= \
                     sum(model.w[j] * max(objs[j]-y_star[j],  self._epsilon)  for j in range(num_objs))
            model.constraints.add(
                constr
            )

        for i in range(len(objs_list_lower)):
            for j in range(num_objs):
                # if y_und[j] is lower than objs_list_lower[i][j], model.b[i, j]=0,
                # otherwise model.b[i, j]=1
                model.constraints.add(
                    (model.y_und[j] - y_star[j]) >=
                    model.b[i, j] * (objs_list_lower[i][j] - y_star[j])# + self._epsilon
                )
            # in order to y_sup dominate objs_list_lower[i],
            # sum_j model.b[i, j] == 0
            # thus sum_j model.b[i, j] >= 1
            model.constraints.add(
                sum(model.b[i, j] for j in range(num_objs)) >= 1
            )
        '''
        for j in range(num_objs):
            model.constraints.add(
                sum(model.b[i, j] for i in range(len(objs_list_lower))) >= 1
            )
        '''

        
        for j in range(num_objs):
            # y_sup[j] >= y_und[j]
            model.constraints.add(
                model.y_sup[j] >= model.y_und[j]
            )
            model.constraints.add(
                model.y_und[j] >= y_star[j]
            )

        model.constraints.add(sum(model.w[j] for j in range(num_objs)) == 1)

        model.obj = pyo.Objective(
            expr=sum(model.w[j] * model.y_sup[j] for j in range(num_objs)) -
                 sum(model.w[j] * model.y_und[j] for j in range(num_objs)),
            sense=pyo.maximize
        )

        solver = appsi.solvers.Gurobi()
        solver.config.mip_gap = self._mip_gap
        solver.config.time_limit = self._time_limit
        solver.solve(model)
        solver.release_license()

        self._weights = np.array([pyo.value(model.w[i]) for i in range(num_objs)])
        y_sup_ = np.array([pyo.value(model.y_sup[i]) for i in range(num_objs)])
        y_und_ = np.array([pyo.value(model.y_und[i]) for i in range(num_objs)])

        # Test if y_und_ dominates any of the objs_list_lower
        for i,objs_lower in enumerate(objs_list_lower):
            if all(y_und_<objs_lower):
                logger.warning("y_und_ dominates an objective lower bound, which is not expected.")
                logger.debug(f"objs_lower: {objs_lower}")
                logger.debug(f"y_und_: {y_und_}")
                logger.debug(" ".join([f"b[{i},{j}]={pyo.value(model.b[i, j])}" for j in range(num_objs)]))

        self._importance = pyo.value(model.obj.expr) if self._weights is not None else 0

        logger.debug(f"Calculated weights: {self._weights}")
        logger.debug(f"Importance: {self._importance}")
        logger.debug(f"Importance: {self._weights@y_sup_ - self._weights@y_und_}")
        logger.debug(f"y_sup: {y_sup_}")
        logger.debug(f"y_und: {y_und_}")
        logger.debug(f"Global lower bound (y*): {y_star}")

    def optimize(self) -> scalar:
        """Optimizes the current solution using the computed weights (as warm start).

        Selects the best solution from the parent set based on the weighted 
        objective value. Then optimizes it with the current weight vector. 
        Stores the resulting model and solution.

        Returns:
            np.ndarray: The optimized solution object.
        """
        best_solution = np.zeros(self._num_objectives)
        best_objective = np.inf

        for solution in self._solutions:
            aux = self.weights@solution.objs
            if aux < best_objective:
                best_objective = aux
                best_solution = solution

        self._solution = copy.copy(best_solution)
        self._solution.optimize(self.weights)

        self.ml_model = self._solution.x
        if np.all(np.equal(self._solution.objs, best_solution.objs)):
           self.best_solution_reached = True
        return self._solution


class Mola:
    """MOLA: Multi-Objective Learning Algorithm
    
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar | None,
        target_gap: float = 0.0,
        target_size: int | None = None,
        red_fact: float = float("inf"),
        smooth_count: int = 0,
        node_time_limit: float = float("inf"),
        node_gap: float = 0.01,
        norm: bool = True,
        utopia_slack: float | str = 1e-5
    ) -> None:
        """Initializes MOLA.

        Args:
            weighted_scalar (scalar): Scalarization object used for weighted-sum optimization.
            single_scalar (scalar | None): Scalarization object used for single-objective optimization.
                If None, only the weighted scalar will be used.
            target_gap (float): Minimum relative gap between solutions to stop refinement.
                Default is 0.0.
            target_size (int | None): Target number of Pareto-optimal solutions to obtain.
                Defaults to 20 × number of objectives if not specified.
            red_fact (float): Reduction factor used in adaptive refinement. Default is infinity.
            smooth_count (int): Number of smoothing iterations before stopping. Default is 0.
            node_time_limit (float): Maximum time allowed (in seconds) per node optimization.
                Default is infinity.
            node_gap (float): Acceptable optimization gap for each node. Default is 0.01.
            norm (bool): Whether to normalize objective vectors before comparison. Default is True.
        """
        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, w_interface))):
            raise ValueError("weighted_scalar and single_scalar must implement the correct interfaces.")

        self._weighted_scalar: scalar = weighted_scalar
        self._single_scalar: scalar = single_scalar
        self._num_objs = self._single_scalar.M
        self._target_gap = target_gap
        self._node_time_limit = node_time_limit
        self._node_gap = node_gap
        self._red_fact = red_fact
        self._norm: bool = norm
        self._max_imp = 1.0
        self._solutions_list = []
        self._ml_models = []
        self.hypervolume_values = []
        self.history_list = []
        self._target_size = target_size or 20 * self._weighted_scalar.M
        self._smooth_count = smooth_count if smooth_count != 0 else (1 if node_time_limit == float('inf') else 5)
        self._utopia_slack = utopia_slack

    def __del__(self) -> None:
        """
        Deletes the solutions list attribute from the object if it exists.
        
        This is a cleanup method called when the object is about to be destroyed.
        It ensures that the `_solutions_list` attribute is deleted if it was created.
        """
        if hasattr(self, '_solutions_list'):
            del self._solutions_list

    @property
    def target_size(self) -> int:
        """Target number of Pareto-optimal solutions.

        Returns:
            int: The number of solutions to aim for in the optimization process.
        """
        return self._target_size

    @property
    def target_gap(self) -> float:
        """Target minimum relative gap between solutions.

        Returns:
            float: The convergence threshold used to stop refinement.
        """
        return self._target_gap

    @property
    def solutions_list(self) -> list[scalar]:
        """List of current Pareto-optimal solutions.

        Returns:
            list[scalar]: An array containing objective values of the solutions.
        """
        return self._solutions_list

    @property
    def curr_imp(self) -> float:
        """Current importance score based on recent iterations.

        Returns:
            float: Maximum importance value from the last `smooth_count` iterations.
        """
        return max(self._importances[-self._smooth_count:])

    @property
    def max_imp(self) -> float:
        """Maximum importance value observed so far.

        Returns:
            float: The highest importance score recorded during optimization.
        """
        return self._max_imp

    @property
    def importances(self) -> list[float]:
        """List of all importance values computed during optimization.

        Returns:
            list[float]: Historical record of importance scores.
        """
        return self._importances
    
    def get_models(self)-> list[Any]:
        """Returns the list of trained machine learning models.

        These models correspond to the solutions generated during optimization.

        Returns:
            list[Any]: A list of ML models associated with each Pareto-optimal solution.
        """
        return self._ml_models

    def get_hypervolumes(self) -> npt.NDArray[np.float64] | list[float]:
        """Returns the hypervolume values computed during optimization.

        Returns:
            npt.NDArray[np.float64] | list[float]: Array of hypervolume values corresponding to each iteration or solution.
        """
        return self.hypervolume_values

    def _grad_squared(self, solution: scalar, obj_index: int) -> float:
        """Computes the squared norm of the gradient for a given objective.

        Args:
            solution (scalar): A solution object containing gradients.
            obj_index (int): Index of the objective function to compute the gradient norm.

        Returns:
            float: Squared Euclidean norm of the gradient vector.
        """
        gradient = solution.gradient[obj_index]
        grad_norm = np.linalg.norm(gradient)

        grad_norm_squared = grad_norm ** 2

        return grad_norm_squared

    def _update_global_lower(self, num_objs: int) -> npt.NDArray[np.float64] | list[float]:
        """Updates the global lower bounds for each objective using Lipschitz-based estimates.

        If gradients are available, estimates a tighter lower bound using a Lipschitz-based formula.
        Otherwise, it falls back to the minimum value of each objective among the solutions.

        Args:
            num_objs (int): Number of objective functions.

        Returns:
            np.ndarray: Updated global lower bounds for each objective.
        """
        objs_lower = np.array([[o for o in p.objs_lower] for p in self._solutions_list]) 
        return objs_lower.min(0)
    
    def inicialization(self) -> WeightSolver:
        """Initializes the optimization process.

        Finds the individual minima of each objective, computes the global lower bounds
        and upper bounds, and builds the first weighted solution. This sets up 
        the optimization for iterative refinement.

        Returns:
            WeightSolver: The first weighted solution used to start the optimization.
        """
        parents = []

        for i in range(self._num_objs):
            logger.debug(f"Finding {i+1}th individual minimum")
            single_scalar = copy.copy(self._single_scalar)
            single_scalar.optimize(i)
            self._solutions_list.append(single_scalar)
            self.history_list.append(single_scalar)
            parents.append(single_scalar)

        self._global_lower = self._update_global_lower(num_objs=self._num_objs)

        first_w_solution = WeightSolver(
            solutions=parents,
            global_lower=self._global_lower,
            #scalarizer=self._weighted_scalar,
        )

        self._max_imp = first_w_solution.importance
        self._importances = [first_w_solution.importance]

        return first_w_solution

    def is_dominated(self, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
        """Checks whether solution `a` is dominated by solution `b`.

        A solution `a` is considered dominated if all its objective values
        are greater than or equal to those of `b`.

        Args:
            a (np.ndarray): Objective values of solution `a`.
            b (np.ndarray): Objective values of solution `b`.

        Returns:
            bool: True if `a` is dominated by `b`, False otherwise.
        """
        return np.all(np.greater_equal(a,b))
    
    def can_add_solution(
        self, 
        new_solution: scalar, 
        solutions: list[scalar]
    ) -> bool:
        """Determines whether a new solution should be added to the Pareto set.

        A solution is added only if it is not dominated by any existing solution.

        Args:
            new_solution (scalar): The candidate solution.
            solutions (list[scalar]): Array of existing solutions.

        Returns:
            bool: True if the new solution is non-dominated and should be added.
        """
        for solution in solutions:
            if self.is_dominated(new_solution.objs, solution.objs):
                return False 
        return True

    def remove_dominated_solutions(
            self,
            new_solution: scalar,
            solutions: list[scalar]
        ) -> list[scalar]:
        """Removes solutions that are dominated by a new one.

        If the new solution is not dominated, it is added to the list, 
        and any existing solutions that it dominates are removed.

        Args:
            new_solution (scalar): The candidate solution to be evaluated.
            solutions (list[scalar]): Existing list of Pareto solutions.

        Returns:
            list[scalar]: Updated list of non-dominated solutions.
        """
        non_dominant_solutions = []

        if self.can_add_solution(new_solution, solutions):
            logger.debug(f"New solution added {new_solution.objs}")
            non_dominant_solutions.append(new_solution)

            for solution in solutions:
                if not self.is_dominated(solution.objs, new_solution.objs):
                    non_dominant_solutions.append(solution)
                else:
                    logger.debug(f"Solution dominated {solution.objs}")
        else:
            non_dominant_solutions = solutions
            logger.debug(f"New solution dominated {new_solution.objs}")

        return non_dominant_solutions

    def update(self, solution: scalar) -> None:
        """Updates the internal solution set with a new candidate.

        Adds the solution to the history, updates the non-dominated 
        set, and recomputes global bounds.

        Args:
            solution (scalar): New solution to be added.
        """
        self.history_list.append(solution)
        if self._solutions_list:
            self._solutions_list = self.remove_dominated_solutions(solution, self._solutions_list)
        else:
            self._solutions_list.append(solution)

        num_objs = len(solution.objs)
        self._global_lower = self._update_global_lower(num_objs=num_objs)

    def _next(self) -> WeightSolver:
        """Computes the next weighted solution.

        Returns:
            WeightSolver: The next weighted solution to be optimized.
        """
        next_w_solution = WeightSolver(
            solutions=self._solutions_list,
            global_lower=self._global_lower,
            #scalarizer=self._weighted_scalar,
            time_limit=self._node_time_limit,
            mip_gap=self._node_gap,
        )
        self._importances.append(next_w_solution.importance)
        return next_w_solution

    def optimize(self) -> None:
        """Runs the full MOLA optimization process.

        Initializes the algorithm, iteratively refines the solution set using
        weighted scalarization, tracks hypervolume improvement, and updates
        the non-dominated front until convergence or target is met.
        """
        start = time.perf_counter()
        next_w_solution = self.inicialization()

        reference_point = np.ones_like(self.solutions_list[0].objs)
        hv = HV(ref_point=reference_point)

        solution_set = []
        iteraction = 0

        while (len(self.solutions_list) < self.target_size
              or len(self.history_list) < self.target_size):

            logger.debug(f"Iteration {iteraction+1}")
            logger.debug(f"Solutions list: {[s.objs for s in self.solutions_list]}")       
            iteraction += 1
            if iteraction >= self.target_size:
                break
            solution = next_w_solution.optimize()
            solution_set.append(solution.objs)
            self.hypervolume_values.append(hv(np.array(solution_set)))

            self._ml_models.append(solution.x)

            if next_w_solution.best_solution_reached:
                logger.info("Best solution found.")
                break

            self.update(solution)
            next_w_solution = self._next()

        self._fit_runtime = time.perf_counter() - start
        logger.info(f"Fit runtime: {self._fit_runtime:.2f} seconds")
