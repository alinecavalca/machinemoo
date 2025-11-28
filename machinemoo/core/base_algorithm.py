import time
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Any, List

from machinemoo.utils.logging_config import get_logger
from machinemoo.scalarization.scalarization_interface import scalar_interface

class BaseMOO(ABC):
    """
    Abstract base class for Multi-Objective Optimization algorithms.
    
    Implements the Template Method pattern for the optimization loop and
    provides utilities for Pareto front maintenance.
    """

    def __init__(
        self, 
        target_size: int, 
        time_limit: float = float('inf'), 
        verbose: bool = False, 
        debug: bool = False
    ) -> None:
        """
        Initialize the base MOO algorithm.

        Args:
            target_size (int): Desired number of solutions.
            time_limit (float): Maximum execution time in seconds.
            verbose (bool): Enable verbose logging.
            debug (bool): Enable debug logging.
        """
        self.target_size = target_size
        self.time_limit = time_limit
        self.logger = get_logger(name=self.__class__.__name__, verbose=verbose, debug=debug)
        
        # Shared state for solutions
        self.solutions_list: List[scalar_interface] = []
        self.history_list: List[scalar_interface] = []  # Stores all solutions found
        self.fit_runtime: float = 0.0

        # Track the number of optimizations performed
        self.n_optimizations: int = 0

    @abstractmethod
    def initialize(self) -> None:
        """Sets up initial solutions, parameters, or first nodes."""
        pass

    @abstractmethod
    def select(self) -> Any:
        """Selects or generates the next node/weight_solver to be optimized."""
        pass

    def is_dominated(self, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
        """
        Checks whether solution `a` is dominated by solution `b`.
        a is dominated by b if all objectives of a >= b.
        """
        return bool(np.all(np.greater_equal(a, b)))

    def _manage_pareto_front(self, new_solution: scalar_interface) -> None:
        """
        Updates the solutions_list to maintain a valid Pareto front.
        
        - Adds new_solution if it is not dominated.
        - Removes existing solutions that are dominated by new_solution.
        """
        # 1. Check if new_solution is dominated by any existing solution
        is_dominated = False
        for sol in self.solutions_list:
            if self.is_dominated(new_solution.objs, sol.objs):
                self.logger.debug(f"New solution dominated by existing solution {sol.objs}")
                is_dominated = True
                break
        
        if is_dominated:
            return

        # 2. If not dominated, add it and remove any solutions it dominates
        self.logger.debug(f"New solution added {new_solution.objs}")
        non_dominated_list = [new_solution]
        
        for sol in self.solutions_list:
            if not self.is_dominated(sol.objs, new_solution.objs):
                non_dominated_list.append(sol)
            else:
                self.logger.debug(f"Existing solution {sol.objs} removed (dominated by new)")
        
        self.solutions_list = non_dominated_list

    def update(self, node: Any, solution: scalar_interface) -> None:
        """
        Updates the internal solution list.
        
        By default, this method:
        1. Adds the solution to the full history.
        2. Filters the 'solutions_list' to keep only non-dominated solutions.
        
        Subclasses can override this to add extra logic (e.g., updating global bounds).
        """
        self.history_list.append(solution)
        self._manage_pareto_front(solution)
    
    def should_stop(self, start_time: float, node: Any) -> bool:
        """Determines if the optimization loop should terminate."""
        if node is None:
            return True
        # Check target size against the filtered Pareto list
        if len(self.solutions_list) >= self.target_size:
            return True
        if (time.perf_counter() - start_time) >= self.time_limit:
            return True
        return False

    def optimize_step(self, node: Any) -> scalar_interface:
        """Executes the optimization on the given node."""
        return node.optimize()

    def optimize(self) -> None:
        """Main optimization loop (Template Method)."""
        start_time = time.perf_counter()
        
        self.logger.info("Starting optimization...")
        self.initialize()

        self.n_optimizations = len(self.solutions_list) # Count initial solutions
        
        node = self.select()
        
        while not self.should_stop(start_time, node):
            solution = self.optimize_step(node)
            self.n_optimizations += 1
            
            if getattr(node, 'best_solution_reached', False):
                self.logger.info("Best solution found.")
                break

            self.update(node, solution)
            node = self.select()

        self.fit_runtime = time.perf_counter() - start_time
        self.logger.info(f"Fit runtime: {self.fit_runtime:.2f} seconds")
        self.logger.info(f"Total optimizations performed: {self.n_optimizations}")
        self.logger.info(f"Final Pareto front size: {len(self.solutions_list)}")