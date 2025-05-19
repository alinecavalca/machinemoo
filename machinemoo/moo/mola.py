import copy
import time
import logging
from typing import List, Optional, Union, Any
#from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import pyomo.environ as pyo
from pyomo.contrib import appsi

from pymoo.indicators.hv import HV

from machinemoo.utils.logging_config import logger
from machinemoo import scalar_interface, w_interface, single_interface

import torch
from sklearn.linear_model import LogisticRegression

__all__ = [
    "Mola"
]

np.random.seed(42) # TODO: Remove

EPS = 1e-10

class WeightSolver:
    def __init__(
        self,
        solutions: List[single_interface],
        global_lower: np.ndarray,
        global_upper: np.ndarray,
        scalarizer: w_interface,
        goal: float = float("inf"),
        time_limit: int = 10,
        mip_gap: float = 0.01,
        norm: bool = False,
        epsilon: float = 0.05
    ) -> None:
        self._scalarizer = scalarizer
        self._num_objectives = solutions[0].M
        self._global_lower = global_lower
        self._solutions = solutions
        self._time_limit = time_limit
        self._mip_gap = mip_gap
        self._epsilon = epsilon

        self.ml_model = None
        self.best_solution_reached = False
        self._compute_weights()

    @property
    def M(self) -> int:
        return self._num_objectives

    @property
    def importance(self) -> float:
        return self._importance

    @property
    def parents(self) -> List[single_interface]:
        return self._solutions

    @property
    def solution(self) -> single_interface:
        return self._solution

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def get_model(self) -> Any:
        return self.ml_model

    def _compute_weights(self) -> None:
        model = pyo.ConcreteModel()
        objs_list = [s.objs for s in self._solutions]
        num_objs = self._num_objectives

        y_star = self._global_lower

        model.b = pyo.Var(range(len(objs_list)), range(num_objs), domain=pyo.Binary)
        model.w = pyo.Var(range(num_objs), domain=pyo.NonNegativeReals)
        model.y_sup = pyo.Var(range(num_objs), domain=pyo.Reals)
        model.y_und = pyo.Var(range(num_objs), domain=pyo.Reals)

        model.constraints = pyo.ConstraintList()

        for objs in objs_list:
            model.constraints.add(
                sum(model.w[j] * model.y_sup[j] for j in range(num_objs)) <= 
                sum(model.w[j] * objs[j] for j in range(num_objs))
            )

        for i in range(len(objs_list)):
            for j in range(num_objs):
                model.constraints.add(
                    (model.y_und[j] - y_star[j]) >=
                    model.b[i, j] * (objs_list[i][j] - y_star[j]) * (1 - self._epsilon)
                )

        for i in range(len(objs_list)):
            model.constraints.add(
                sum(model.b[i, j] for j in range(num_objs)) >= num_objs - 1
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
        self._importance = pyo.value(model.obj.expr) if self._weights is not None else 0

        logger.debug(f"Calculated weights: {self._weights}")
        logger.debug(f"Importance: {self._importance}")
        logger.debug(f"Global lower bound (y*): {y_star}")

    def optimize(self):
        """
        Optimizes the solution using the best solution as a warm start.

        Iterates through the list of solutions to find the best solution based on the weighted objective function.
        Then, it optimizes the best solution using the provided weights and updates the internal solution.

        Returns:
             The optimized solution.
        """
        #usar isso no random
        best_solution = None
        best_objective = np.inf

        for solution in self._solutions:
            aux = self.weights@solution.objs
            if aux < best_objective:
                best_objective = aux
                best_solution = solution

        self._solution = copy.deepcopy(best_solution)
        self._solution.optimize(self.weights)
        #self._solution = copy.copy(self._scalarizer)
        #self._solution.optimize(self.weights)
        self.ml_model = self._solution.x
        if np.all(np.equal(self._solution.objs, best_solution.objs)):
           self.best_solution_reached = True
        return self._solution

    #def optimize(self) -> single_interface:
    #    best = min(self._solutions, key=lambda s: self.weights @ s.objs)
    #    self._solution = copy.copy(best)
    #    self._solution.optimize(self.weights)
    #    self.ml_model = self._solution.x
    #    if np.all(np.equal(self._solution.objs, best.objs)):
    #        self.best_solution_reached = False
    #    return self._solution

class Mola:
    def __init__(
        self,
        weighted_scalar: Union[scalar_interface, w_interface],
        single_scalar: Union[scalar_interface, single_interface],
        target_gap: float = 0.0,
        target_size: Optional[int] = None,
        red_fact: float = float("inf"),
        smooth_count: Optional[int] = None,
        node_time_limit: float = float("inf"),
        node_gap: float = 0.01,
        norm: bool = True
    ) -> None:
        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, w_interface))):
            raise ValueError("weighted_scalar and single_scalar must implement the correct interfaces.")

        self._weighted_scalar = weighted_scalar
        self._single_scalar = single_scalar
        self._target_gap = target_gap
        self._node_time_limit = node_time_limit
        self._node_gap = node_gap
        self._red_fact = red_fact
        self._norm = norm
        self._max_imp = 1.0
        self._solutions_list = []
        self._ml_models = []
        self.hypervolume_values = []
        self.history_list = []
        self._target_size = target_size or 20 * self._weighted_scalar.M
        self._smooth_count = smooth_count if smooth_count is not None else (1 if node_time_limit == float('inf') else 5)

    def __del__(self) -> None:
        """
        Deletes the _solutions list attribute from the object if it exists.
        """
        if hasattr(self, '_solutions_list'):
            del self._solutions_list

    @property
    def target_size(self) -> int:
        return self._target_size

    @property
    def target_gap(self) -> float:
        return self._target_gap

    @property
    def solutions_list(self) -> List:
        return self._solutions_list

    @property
    def curr_imp(self) -> float:
        return max(self._importances[-self._smooth_count:])

    @property
    def max_imp(self) -> float:
        return self._max_imp

    @property
    def importances(self) -> List[float]:
        return self._importances
    
    def get_models(self):
        return self._ml_models

    def get_hypervolumes(self) -> List[float]:
        return self.hypervolume_values

    def grad_squared(self, solution, obj_index):
        gradient = solution.gradient[obj_index]
        grad_norm = np.linalg.norm(gradient)

        grad_norm_squared = grad_norm ** 2

        return grad_norm_squared

    def _update_global_lower(self, num_objs):
        L = 1/4 # Lipschitz constante
        if hasattr(self._solutions_list[0].objs, "gradient"):
            adjusted_lower = np.zeros(num_objs)
            for obj_index in range(num_objs):
                adjusted_min = min(s.objs[obj_index] - ((1 / (2 * L)) * self.grad_squared(s, obj_index))
                          for s in self._solutions_list)
                adjusted_lower[obj_index] = adjusted_min
            logger.info(f"[Lipschitz] Updated global lower bounds: {adjusted_lower}")
        else:
            objs = np.array([[o for o in p.objs] for p in self._solutions_list])
            adjusted_lower = objs.min(0)
        return adjusted_lower
    
    def inicialization(self):
        """
        Initializes the optimization process by finding individual minima, 
        calculating global bounds, and determining the first weighted solution.

        Parameters:
            None

        Returns:
            The first weighted solution (first_wsol)
        """
        self._M = self._single_scalar.M
        parents = []

        for i in range(self._M):
            logger.debug(f"Finding {i+1}th individual minimum")
            single_scalar = copy.copy(self._single_scalar)
            single_scalar.optimize(i)
            self._solutions_list.append(single_scalar)
            self.history_list.append(single_scalar)
            parents.append(single_scalar)

        objs = np.array([[o for o in p.objs] for p in parents])
        num_objs = len(objs[0])
        self._global_lower = self._update_global_lower(num_objs=num_objs)
        #self._global_lower = objs.min(0)
        #self._global_lower = np.zeros(shape=num_objs)
        self._global_upper = objs.max(axis=0)

        first_w_solution = WeightSolver(
            parents,
            self._global_lower,
            self._global_upper,
            self._weighted_scalar,
            norm=self._norm
        )

        self._max_imp = first_w_solution.importance
        self._importances = [first_w_solution.importance]

        return first_w_solution

    def is_dominated(self, a, b):
        return np.all(np.greater_equal(a,b))
    
    def can_add_solution(self, new_solution, solutions):
        for solution in solutions:
            if self.is_dominated(new_solution.objs, solution.objs):
                return False 
        return True

    def remove_dominated_solutions(self, new_solution, solutions):
        non_dominant_solutions = []

        if self.can_add_solution(new_solution, solutions):
            print("new solution added", new_solution.objs)
            non_dominant_solutions.append(new_solution)

            for solution in solutions:
                if not self.is_dominated(solution.objs, new_solution.objs):
                    non_dominant_solutions.append(solution)
                else:
                    print("solution dominated", solution.objs)
        else:
            non_dominant_solutions = solutions
            print("new solution dominated", new_solution.objs)

        return non_dominant_solutions

    def update(self, solution) -> None:
        self.history_list.append(solution)
        if self._solutions_list:
            self._solutions_list = self.remove_dominated_solutions(solution, self._solutions_list)
        else:
            self._solutions_list.append(solution)

        objs = np.array([[o for o in s.objs] for s in self._solutions_list])
        num_objs = len(solution.objs)
        self._global_lower = self._update_global_lower(num_objs=num_objs)
        #self._global_lower = np.zeros(shape=num_objs)
        #self._global_lower = objs.min(0)
        self._global_upper = objs.max(axis=0)

    def _next(self):
        next_w_solution = WeightSolver(
            self._solutions_list,
            self._global_lower,
            self._global_upper,
            self._weighted_scalar,
            goal=self.curr_imp * self._red_fact,
            time_limit=self._node_time_limit,
            mip_gap=self._node_gap,
            norm=self._norm
        )
        self._importances.append(next_w_solution.importance)
        return next_w_solution

    def plot_mu(self) -> None:
        plt.plot(self._importances, color="purple", linewidth=2)
        plt.xlabel("Iterations")
        plt.ylabel("mu")
        plt.title("Evolution of margin over iterations")
        plt.grid()
        plt.show()

    def optimize(self) -> None:
        start = time.perf_counter()
        next_w_solution = self.inicialization()

        reference_point = np.ones_like(self.solutions_list[0].objs)
        hv = HV(ref_point=reference_point)

        solution_set = []
        count = 0

        while (#self.curr_imp / self._max_imp > self._target_gap and
               len(self.solutions_list) < self._target_size
               or len(self.history_list) < self._target_size
               ):

            logger.debug(f"Iteration {count+1}")
            logger.debug(f"Solutions list: {[s.objs for s in self.solutions_list]}")
            if count == 150:
                break
            count += 1

            solution = next_w_solution.optimize()
            solution_set.append(solution.objs)
            self.hypervolume_values.append(hv(np.array(solution_set)))

            ml_model = next_w_solution.get_model()
            self._ml_models.append(solution.x)

            if next_w_solution.best_solution_reached:
                logger.info("Best solution found.")
                break

            self.update(solution)
            next_w_solution = self._next()

        self.plot_mu()
        self._fit_runtime = time.perf_counter() - start
        logger.info(f"Fit runtime: {self._fit_runtime:.2f} seconds")
