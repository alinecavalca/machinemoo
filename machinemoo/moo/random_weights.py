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

# License: BSD 3 clause

import copy
import time
import logging
import numpy as np
import numpy.typing as npt

from machinemoo.utils.typing import scalar
from machinemoo.utils.logging_config import logger
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "random_weights"
]

class weight_iter():
    """Solves a scalarization weight optimization problem for multi-objective learning.

    This class normalizes the given weight vector if requested, applies it to a 
    scalarization problem, and keeps track of the associated solution.
    """
    def __init__(
        self,
        w: npt.NDArray[np.float64],
        globalL: npt.NDArray[np.float64],
        globalU: npt.NDArray[np.float64],
        weightedScalar: scalar,
        solutions: list[scalar],
        norm: bool = True
    ) -> None:
        """Initialize the weight solver for multi-objective optimization.

        Args:
        w (npt.NDArray[np.float64]): Initial weight vector, of shape (M,).
        globalL (npt.NDArray[np.float64]): Lower bounds for each objective of shape (M,).
        globalU (npt.NDArray[np.float64]): Upper bounds for each objective of shape (M,).
        weightedScalar (scalar): Scalarization object used to optimize with the new weight vector.
        solutions (list[scalar]): List of scalarized solutions representing the current approximation of the Pareto frontier.
        norm (bool): Whether to normalize the weight vector and objective values. Defaults to True.
        
        """
        self.__weightedScalar = weightedScalar
        self.__globalL, self.__globalU = globalL, globalU
        self.__norm = norm
        self.__solutions = solutions
        self.best_solution_reached = False
        self.__w = self.__calcW(w)

    @property
    def w(self) ->  npt.NDArray[np.float64]:
        """The weight vector obtained from the optimization.

        Returns:
            np.ndarray: The vector of weights for scalarization.
        """
        return self.__w

    def optimize(self) -> scalar:
        """Optimizes the current solution using the computed weights (as warm start).

        Selects the best solution from the parent set based on the weighted 
        objective value. Then optimizes it with the current weight vector. 

        Returns:
            np.ndarray: The optimized solution object.
        """
        #self.__solution = copy.copy(self.__weightedScalar)
        #self.__solution.optimize(self.w)
        best_solution = None
        best_objective = np.inf

        for solution in self.__solutions:
            aux = self.w@solution.objs
            if aux < best_objective:
                best_objective = aux
                best_solution = solution

        self.__solution = copy.copy(best_solution)
        self.__solution.optimize(self.w)

        if np.all(np.equal(self.__solution.objs, best_solution.objs)):
           self.best_solution_reached = True
        return self.__solution

    def __calcW(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize the weights

        Args:
            w (np.ndarray): Weighting vector, ponderates the objectives of the
                            weighted sum method.

        Returns:
            np.ndarray: Normalized weighting vector if normalization is True
        """
        if self.__norm:
            w = w/(self.__globalU-self.__globalL)

        return w


class random_weights():
    """A posteriori multi-objetive optimization algorithm based sampling random
    points in the simplex as weight vectors.

    This method approximates the Pareto frontier by solving multiple
    scalarizations using randomly generated weights.
    """
    def __init__(
        self,
        weightedScalar: scalar = None,
        singleScalar: scalar = None,
        targetSize: int | None = None,
        norm: bool = True
    ) -> None:
        """Initialize the random weight sampling optimizer.

        Args:
        weightedScalar (scalar): An instance of a class solving the weighted scalarization
        singleScalar (scalar): An instance of a class solving single-objective problems.
        targetSize (int): Desired number of Pareto solutions.
            Defaults to 20 × number of objectives if not specified.
        norm (bool): Whether to normalize objective vectors before comparison. Default is True.
        
        """
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
        if (not isinstance(weightedScalar, scalar_interface) or
            not isinstance(weightedScalar, w_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError('weightedScalar' + ' and ' + 'singleScalar' +
                             'must be a mo_problem implementation.')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm

        self.__solutionsList = []
        self.__candidatesList = {}

    def __del__(self) -> None:
        """
        Deletes the solutions list attribute from the object if it exists.
        
        This is a cleanup method called when the object is about to be destroyed.
        """
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def solutionsList(self) -> list[scalar]:
        """List of current Pareto-optimal solutions.

        Returns:
            list[scalar]: An array containing objective values of the solutions.
        """
        return self.__solutionsList

    @property
    def targetSize(self) -> int:
        """Target number of Pareto-optimal solutions.

        Returns:
            int: The number of solutions to aim for in the optimization process.
        """
        return self.__targetSize

    def inicialization(self) -> None:
        """Initializes the optimization process.

        Finds the individual minima of each objective, computes the global lower 
        and upper bounds.
        """
        self.__M = self.__singleScalar.M
        neigO = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i+1)+'th individual minima')
            singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

    def update(self, solution: scalar) -> None:
        """Updates the internal solution set with a new candidate.

        Args:
            solution (scalar): New solution to be added.
        """
        self.__solutionsList.append(solution)
        logger.debug(str(len(self.solutionsList))+'th solution')

    def select(self) -> weight_iter:
        """Selects the next scalarization using a randomly sampled
        weight vector from the unit simplex.

        The sampling follows the method proposed in Smith (2009), which ensures
        a uniform distribution over the simplex.

        Returns:
            weight_iter: A scalarization object initialized with the sampled weight.
        """
        rnd = np.array(sorted([0] +
                              [np.random.rand() for i in range(self.__M-1)] +
                              [1]))
        w = np.array([rnd[i+1]-rnd[i] for i in range(self.__M)])
        w = w/w.sum()
        return weight_iter(w, self.__globalL, self.__globalU,
                           self.__weightedScalar, self.__solutionsList, norm=self.__norm)

    def optimize(self) -> None:
        """Runs the full MONISE optimization process.

        Runs the optimization loop until the desired number of solutions
        is obtained or the best solution is reached.

        Repeatedly samples a random weight vector, solves the scalarization,
        and stores the result.
        """
        start = time.perf_counter()
        self.inicialization()

        node = self.select()

        while (node is not None and
               len(self.solutionsList) < self.targetSize):

            solution = node.optimize()
            if node.best_solution_reached:
                logger.info("Best solution found.")
                break

            self.update(solution)
            node = self.select()
        self.__fit_runtime = time.perf_counter() - start