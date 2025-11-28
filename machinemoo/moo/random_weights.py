# -*- coding: utf-8 -*-
"""
A posteriori multiobjective optimization method based on
random weighted sum method with Smith simplex sampling.

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Smith, N. and Tromble, R.
    Sampling Uniformly from the Unit Simplex Naïve Algorithms
    2004
"""

import copy
import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, cast

from machinemoo.core.base_algorithm import BaseMOO
from machinemoo.utils.typing import scalar
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "RandomWeights"
]

class WeightNode:
    """
    Helper class to manage weight vectors and scalarization solving for RandomWeights.
    """
    def __init__(
        self,
        w: npt.NDArray[np.float64],
        global_lower: npt.NDArray[np.float64],
        global_upper: npt.NDArray[np.float64],
        weighted_scalar: scalar,
        solutions: List[scalar],
    ) -> None:
        """
        Args:
            w (npt.NDArray[np.float64]): Initial weight vector.
            global_lower (npt.NDArray[np.float64]): Lower bounds (utopia point).
            global_upper (npt.NDArray[np.float64]): Upper bounds (nadir point).
            weighted_scalar (scalar): Scalarization object.
            solutions (List[scalar]): Current list of solutions (for warm start).
        """
        self._weighted_scalar = weighted_scalar
        self._global_lower = global_lower
        self._global_upper = global_upper
        self._solutions = solutions
        self.best_solution_reached = False
        
        self._w = self._calc_w(w)
        self._solution: Optional[scalar] = None

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """The weight vector obtained from the optimization."""
        return self._w

    def optimize(self) -> scalar:
        """
        Optimizes the current solution using the computed weights.
        Attempts to select the best existing solution as a warm start.
        """
        best_solution = None
        best_objective = np.inf

        # Find best warm start from existing solutions
        for solution in self._solutions:
            # Check dimensions to avoid broadcasting errors
            if len(solution.objs) == len(self.w):
                val = self.w @ solution.objs
                if val < best_objective:
                    best_objective = val
                    best_solution = solution

        # Setup and run scalarization
        self._solution = copy.copy(self._weighted_scalar)
        
        # If we found a warm start, we could potentially set it here 
        # (dependent on scalarizer implementation support for x_init)
        # For now, we just proceed to optimize.
        
        self._solution.optimize(self.w)

        # Check if we just retrieved an existing solution (convergence check)
        if best_solution is not None and np.all(np.equal(self._solution.objs, best_solution.objs)):
           self.best_solution_reached = True
           
        return self._solution

    def _calc_w(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Normalizes the weights based on the objective ranges if normalization is enabled.
        """            
        return w


class RandomWeights(BaseMOO):
    """
    A posteriori multi-objective optimization algorithm based on sampling 
    random points in the simplex as weight vectors.
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar,
        target_size: int = 50,
        time_limit: float = float('inf'),
        verbose: bool = False,
        debug: bool = False,
        **kwargs: Any # Catch-all for compatibility
    ) -> None:
        """
        Initialize the RandomWeights optimizer.

        Args:
            weighted_scalar (scalar): Scalarization object for weighted sum.
            single_scalar (scalar): Scalarization object for single objectives.
            target_size (int): Target number of Pareto-optimal solutions.
            time_limit (float): Maximum execution time.
            verbose (bool): Enable verbose logging.
            debug (bool): Enable debug logging.
        """
        super().__init__(target_size=target_size, time_limit=time_limit, verbose=verbose, debug=debug)
        
        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, single_interface))):
            raise ValueError("weighted_scalar and single_scalar must be valid scalarization implementations.")

        self._weighted_scalar = weighted_scalar
        self._single_scalar = single_scalar
        
        # State variables initialized in initialize()
        self._M: int = 0
        self._global_lower: Optional[npt.NDArray[np.float64]] = None
        self._global_upper: Optional[npt.NDArray[np.float64]] = None

    def initialize(self) -> None:
        """
        Initializes the optimization by finding individual minima for each objective.
        """
        self._M = self._single_scalar.M
        neig_o = []
        
        # 1. Find individual minima (Extreme points)
        for i in range(self._M):
            single_s = copy.copy(self._single_scalar)
            self.logger.debug(f"Finding {i+1}th individual minima")
            single_s.optimize(i)
            neig_o.append(single_s.objs)
            self.solutions_list.append(single_s)

        # 2. Calculate Utopia (lower) and Nadir (upper) approximations
        neig_o_arr = np.array(neig_o)
        self._global_lower = neig_o_arr.min(0)
        self._global_upper = neig_o_arr.max(0)

    def select(self) -> WeightNode:
        """
        Selects the next scalarization using a randomly sampled weight vector 
        from the unit simplex.
        """
        # Smith (2004) method for uniform sampling on simplex
        rnd = np.array(sorted([0] + [np.random.rand() for _ in range(self._M - 1)] + [1]))
        w = np.array([rnd[i+1] - rnd[i] for i in range(self._M)])
        
        # Ensure w sums to 1 (it should by definition, but good for stability)
        w = w / w.sum()
        
        # Ensure initialize() has been called and bounds are available
        if self._global_lower is None or self._global_upper is None:
            raise RuntimeError("RandomWeights.select() called before initialize(); global bounds not set.")
        # Cast to non-Optional for the WeightNode constructor
        global_lower = cast(npt.NDArray[np.float64], self._global_lower)
        global_upper = cast(npt.NDArray[np.float64], self._global_upper)
        return WeightNode(
            w,
            global_lower,
            global_upper,
            self._weighted_scalar,
            self.solutions_list,
        )

    def update(self, node: Any, solution: scalar) -> None:
        """
        Updates the internal solution set using the BaseMOO logic
        to filter out dominated solutions.
        """
        # Explicitly call base class to ensure non-domination filtering happens
        super().update(node, solution)