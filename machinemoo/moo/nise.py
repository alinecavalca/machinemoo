# -*- coding: utf-8 -*-
"""
Noninferior Set Estimation implementation

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Cohon, Jared L., Church, Richard L., Sheer, Daniel P.
    Generating multiobjective trade‐offs: An algorithm for bicriterion problems
    1979
    Water Resources Research
"""
# License: BSD 3 clause

import copy
import time
import bisect
import numpy as np
import numpy.typing as npt
import warnings

from machinemoo.utils.typing import scalar
from machinemoo import get_logger
from machinemoo import scalar_interface, w_interface, single_interface

__all__ = [
    "nise"
]

logger = get_logger(f"moo.{__name__}")

class wNode():
    """Solves a scalarization weight optimization problem for multi-objective learning.

    Node used in the NISE algorithm to represent a candidate region
    for multi-objective optimization via weighted scalarization.
    """
    def __init__(
        self,
        parents: list[scalar],
        globalL: npt.NDArray[np.float64],
        globalU: npt.NDArray[np.float64],
        weightedScalar: scalar,
        distance: str = 'l2',
        norm: bool = True
    ) -> None:
        """Initializes the wNode.

        Args:
            parents (list[scalar]): Parent solutions used to derive this node.
            globalL (np.ndarray): Global lower bounds of the objectives.
            globalU (np.ndarray): Global upper bounds of the objectives.
            weightedScalar (scalar): Scalarization function used for optimization.
            norm (bool): Whether to normalize objective vectors (unused here). Default is False.
            distance (str): Distance metric to compute node importance. Default is 'l2'.
        """
        self.__distance = distance
        self.__weightedScalar = weightedScalar
        self.__M = weightedScalar.M
        self.__globalL, self.__globalU = globalL, globalU
        self.__parents = parents
        self.__norm = norm
        self.__calcW()
        self.__calcImportance()

    @property
    def importance(self) -> float:
        """Importance score of this weight vector for next iteration.

        Returns:
            float: The separation margin between upper and lower bounds.
        """
        return self.__importance

    @property
    def parents(self) -> list[scalar]:
        """List of parent solutions used to generate this node.

        Returns:
            list[scalar]: Array of solution objects used as input.
        """
        return self.__parents

    @property
    def solution(self) -> scalar:
        """Returns the scalarized solution for this node.

        Returns:
            scalar: The best solution found using the computed weights.
        """
        return self.__solution

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """ Weighting vector, which ponderates the objectives of the
        weighted sum method in the scalarization method.

        Returns:
            np.ndarray: The vector of weights for scalarization.
        """
        return self.__w

    def __normf(self, obj: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize the objectives

        Args:
            objs (np.ndarray): Objective vector to be normalized
        
        Returns:
            np.ndarray: Normalized objective vector
        """
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Normalize the weights

        Args:
            w (np.ndarray): Weighting vector, ponderates the objectives of the
                            weighted sum method.

        Returns:
            np.ndarray: Normalized weighting vector if normalization is True
        """
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/w_.sum()
        else:
            return w

    @property
    def useful(self) -> bool:
        """Check if the current solution provides new information between parents.
        
        Returns:
            bool: True if current solution provedes new information, False otherwise
        """
        P = np.array([[i for i in p.objs] for p in self.parents])
        between = ((self.__solution.objs >= P.min(axis=0)).all()
                   and (self.__solution.objs <= P.max(axis=0)).any())
        equal = ((self.__solution.objs == P[0, :]).all() or
                 (self.__solution.objs == P[1, :]).all())
        return between and not equal

    def optimize(self, hotstart: list[scalar] = []) -> scalar:
        """Optimize using the weighted scalar method.

        Args:
            hotstart (list[scalar]): Initial solution.

        Returns:
            scalar: Optimized solution for this weight vector.
        """
        self.__solution = copy.copy(self.__weightedScalar)
        
        self.__solution.optimize(self.w)
            
        return self.__solution

    def __calcImportance(self) -> None:
        """Calculate the importance of the node based on the objective geometry."""
        if self.__w is None:
            self.__importance = 0
        else:
            X = [[i for i in self.__normw(p.w)] for p in self.__parents]
            y = [self.__normf(p.objs)@self.__normw(p.w) for p in self.__parents]
    
            r = self.__normf(self.__parents[0].objs)
            p = np.linalg.solve(X, y)
            if self.__distance == 'l2':
                self.__importance = (self.__normw(self.w)@(r-p) /
                                     np.linalg.norm(self.__normw(self.w)))**2
            else:
                self.__importance = self.__normw(self.w)@(r-p)

    def __calcW(self) -> None:
        """Solve linear system to compute new weighting vector."""
        objs = [i.objs for i in self.__parents]
        logger.debug(objs)
        X = [[i for i in self.__normf(p.objs)]+[-1] for p in self.__parents]
        X = np.array(X + [[1]*self.__M+[0]])
        y = [0]*self.__M+[1]

        try:
            w_ = np.linalg.solve(X, y)[:self.__M]
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)

            self.__w = w_/w_.sum()
        except np.linalg.LinAlgError:
            self.__w = None

class nise():
    """
    Non-inferior Set Estimation (NISE) algorithm for multi-objective optimization.

    This algorithm is based on weighted sum and iteratively explores the Pareto
    frontier by solving a sequence of weighted scalarization problems using the
    wNode structure.
    """
    def __init__(
        self,
        weightedScalar: scalar,
        singleScalar: scalar | None= None,
        targetGap: float = 0.0,
        targetSize: int | None = None, 
        hotstart: list[scalar] = [],
        norm: bool = True, 
        timeLimit: float = float('inf'),
        objective: str = 'l2'
    ) -> None:
        """Initializes NISE class

        Args:
            weightedScalar (scalar): An instance of a class solving weighted sum scalarizations.
            singleScalar (scalar | None): An instance of a class solving single-objective problems.
            targetGap (float): Termination criterion based on importance ratio. Default is 0.0.
            targetSize (int): Desired number of Pareto solutions.
                Defaults to 20 × number of objectives if not specified.
            hotstart (list[scalar]): Initial solutions/models to warm-start the optimization.
            norm (bool): Whether to normalize objectives and weights. Default is True.
            timeLimit (float): Maximum execution time. Default is infinity.
            objective (str): Distance metric to compute node importance. Default is 'l2'.
        """
        if (not isinstance(weightedScalar, scalar_interface) or
            not isinstance(weightedScalar, w_interface) or
            not isinstance(singleScalar, scalar_interface) or
                not isinstance(singleScalar, single_interface)):
            raise ValueError('''weightedScalar and singleScalar must be a
                             mo_problem implementation.''')

        self.__weightedScalar = weightedScalar
        self.__singleScalar = singleScalar
        self.__targetGap = targetGap
        self.__targetSize = (targetSize if targetSize is not None else
                             20*self.__weightedScalar.M)
        self.__norm = norm

        self.__currImp = 1
        self.__maxImp = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []
        self.__timeLimit = timeLimit
        self.__objective = objective

    def __del__(self) -> None:
        """Deletes the solutions list attribute from the object if it exists.
        
        This is a cleanup method called when the object is about to be destroyed.
        """
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def targetSize(self) -> int: 
        """Target number of Pareto-optimal solutions.

        Returns:
            int: The number of solutions to aim for in the optimization process.
        """
        return self.__targetSize

    @property
    def targetGap(self) -> float:
        """Target minimum relative gap between solutions.

        Returns:
            float: The convergence threshold used to stop refinement.
        """
        return self.__targetGap

    @property
    def maxImp(self) -> float:
        """Maximum importance value observed so far.

        Returns:
            float: The highest importance score recorded during optimization.
        """
        return self.__maxImp

    @property
    def currImp(self) ->  float:
        """Current importance score based on recent iterations.

        Returns:
            float: Maximum importance value from the last `smooth_count` iterations.
        """
        return self.__currImp

    @property
    def solutionsList(self) -> list[scalar]:
        """List of current Pareto-optimal solutions.

        Returns:
            list[scalar]: An array containing objective values of the solutions.
        """
        return self.__solutionsList

    @property
    def hotstart(self) -> list[scalar]:
        """Get the list of initial solutions (hotstart) for the optimization process.

        Returns:
            list[scalar]: Combined list of warm-start solutions and previously found solutions.
        """
        return self.__hotstart+self.solutionsList

    def inicialization(self) -> None:
        """Initialize scalarizations and compute extreme points (utopia/nadir).

        Raises:
            ValueError: If number of objectives is not 2.
        """
        self.__M = self.__singleScalar.M
        if self.__M != 2:
            raise ValueError('''NISE only support MOO problems with
                             2 objectives.''')
        neigO = []
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
            neigO.append(singleS.objs)
            self.__solutionsList.append(singleS)
            parents.append(singleS)

        neigO = np.array(neigO)
        self.__globalL = neigO.min(0)
        self.__globalU = neigO.max(0)

        self.__candidatesList = wNode(parents, self.__globalL, self.__globalU,
                                       self.__weightedScalar, norm=self.__norm,
                                       distance=self.__objective)
        self.__candidatesList = [self.__candidatesList]

        self.__maxImp = self.__candidatesList[-1].importance
        self.__currImp = self.__candidatesList[-1].importance

    def select(self) ->  wNode | None:
        """Selects next candidate node to explore.

        Returns:
            wNode or None: The most relevant unbounded candidate.
        """
        bounded_ = True
        while bounded_ and self.__candidatesList != []:
            candidate = self.__candidatesList.pop()
            bounded_ = (candidate.w < 0).any()

        if bounded_:
            return None
        else:
            return candidate

    def update(self, node: wNode, solution: scalar) -> None:
        """
        Update solutions list with a new solution.
        Add a new solution to the Pareto set and explore new regions.

        Args:
            node (wNode): Current candidate node.
            solution (scalar): Corresponding scalarized solution.
        """
        try:
            self.solutionsList.append(solution)
            if any([all(p.objs==node.solution.objs) for p in node.parents]):
                raise RuntimeError('Optimization issues.')
            if not node.useful:
                raise RuntimeError('Optimization issues.')
            self.__branch(node, solution)
        except RuntimeError:# as msg:
            warnings.warn('Not optimal solver or nonconvex problem')

        if self.__candidatesList != []:
            self.__currImp = self.__candidatesList[-1].importance
        gap = self.currImp/self.__maxImp

        logger.debug(str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap))

    def __branch(self, node: wNode, solution: scalar) -> None:
        """
        Generate new candidate nodes by branching from a given solution.

        Args:
            node (wNode): Parent node.
            solution (scalar): New solution to create branches from.
        """
        for i in range(self.__M):
            parents = [p if j != i else node.solution
                       for j, p in enumerate(node.parents)]
            boxW = wNode(parents, self.__globalL, self.__globalU,
                          self.__weightedScalar, norm=self.__norm, 
                          distance=self.__objective)

            # avoiding over representation of some regions
            maxdist = max(abs(parents[0].objs-parents[1].objs)/(self.__globalU-self.__globalL))
            
            if boxW.w is not None and not (boxW.w < 0).any() and maxdist>1./self.targetSize:
                index = bisect.bisect_left([c.importance
                                            for c in self.__candidatesList],
                                           boxW.importance)
                self.__candidatesList.insert(index, boxW)

    def optimize(self) -> None:
        """
        Execute the full NISE algorithm to approximate the Pareto frontier.
        """
        start = time.perf_counter()
        self.inicialization()

        node = self.select()

        while (node is not None and
               self.currImp/self.maxImp > self.targetGap and
               len(self.solutionsList) < self.targetSize and
               time.perf_counter()-start<self.__timeLimit):

            solution = node.optimize(hotstart=self.hotstart)
            self.update(node, solution)
            node = self.select()
        self.__fit_runtime = time.perf_counter() - start
        logger.info(f"Fit runtime: {self.__fit_runtime:.2f} seconds")
