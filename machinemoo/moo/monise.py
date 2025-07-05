# -*- coding: utf-8 -*-
"""
Many Objective Noninferior Estimation

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Raimundo, Marcos M.
    MONISE - Many Objective Noninferior Estimation
    2017
    arXiv
"""
# License: BSD 3 clause
import mip
import copy
import time
import logging
import numpy as np
import numpy.typing as npt

from machinemoo.utils.typing import scalar
from machinemoo.utils.logging_config import logger
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "monise"
]

#MAXINT = 200000000000000
MAXINT = 2000000000

class weight_solv():
    """Solves a scalarization weight optimization problem for multi-objective learning.
    
    This class estimates a new weighting vector based on an existing
    list of scalarized solutions.
    """
    def __init__(
        self,
        solutionsList: list[scalar],
        globalL: npt.NDArray[np.float64],
        globalU: npt.NDArray[np.float64],
        weightedScalar: scalar,
        goal: float = float('inf'),
        time_limit: float = 10.0,
        mip_gap: float = 0.01, 
        norm: bool = False
    ) -> None:
        """Initialize the weight solver for multi-objective optimization.

        Args:
            solutionsList (list[scalar]): 
                List of scalarized solutions representing the current approximation of the Pareto frontier.
            globalL (npt.NDArray[np.float64]): 
                Lower bounds for each objective (typically the utopia point).
            globalU (npt.NDArray[np.float64]): 
                Upper bounds for each objective (typically the nadir point).
            weightedScalar (scalar): 
                Scalarization object used to optimize with the new weight vector.
            goal (float): 
                Target importance value to reach. Defaults to infinity.
            time_limit (float): 
                Maximum time (in seconds) allowed for solving the internal optimization problem. Defaults to 10.0.
            mip_gap (float): 
                Acceptable optimality gap for solving the internal mixed-integer programming problem. Defaults to 0.01.
            norm (bool): 
                Whether to normalize the objective space based on `globalL` and `globalU`. Defaults to False.
        """
        self.__weightedScalar = weightedScalar
        self.__M = solutionsList[0].M
        self.__globalL, self.__globalU = globalL, globalU
        self.solutionsList = solutionsList
        self.__goal = goal
        self.__time_limit = time_limit
        self.__mip_gap = mip_gap
        self.__norm = norm
        self.best_solution_reached = False
        if len(self.solutionsList) == self.M:
            self.__calcFirstW()
        else:
            self.__calcW(goal=goal)

    @property
    def M(self) -> int:
        """Number of objective functions.

        Returns:
            int: The number of objectives.
        """
        return self.__M

    @property
    def importance(self) -> float:
        """Importance score computed from the separation margin optimization.

        Returns:
            float: The separation margin between upper and lower bounds.
        """
        return self.__importance

    #@property
    #def parents(self):
    #    return self.__parents

    @property
    def solution(self) -> scalar:
        """The current optimized solution.

        Returns:
            scalar: The best solution found using the computed weights.
        """
        return self.__solution

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """The weight vector obtained from the MILP optimization.

        Returns:
            np.ndarray: The vector of weights for scalarization.
        """
        return self.__w

    def optimize(self, hotstart: list[scalar] = []) -> scalar:
        """Optimize using the weighted scalar method.

        Args:
            hotstart (list[scalar]): Initial solution.

        Returns:
            scalar: Optimized solution for this weight vector.
        """
        #self.__solution = copy.copy(self.__weightedScalar)
        #try:
        #    self.__solution.optimize(self.w, [])
        #except:
        #    self.__solution.optimize(self.w)
        #return self.__solution
        best_solution = np.zeros(self.M)
        best_objective = np.inf

        for solution in self.solutionsList:
            aux = self.w@solution.objs
            if aux < best_objective:
                best_objective = aux
                best_solution = solution

        self.__solution = copy.copy(best_solution)
        self.__solution.optimize(self.w)
        self.ml_model = self.__solution.x
        if np.all(np.equal(self.__solution.objs, best_solution.objs)):
           self.best_solution_reached = True
        return self.__solution

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

    def __calcD(
        self,
        solT: scalar,
        solList: list[scalar] | None = None
    ) -> tuple[float, list[float]]:
        """Compute the maximum normalized objective-wise distance between the
        target solution and a list of reference solutions.

        Args:
            solT (scalar): The target solution to compare against.
            solList (list[scalar], optional): List of solutions for comparison.
                If None, defaults to `self.solutionsList`.

        Returns:
            tuple[float, list[float]]: A tuple containing:
                - The maximum distance (float) between `solT` and the list.
                - A list of individual distances (list[float]) for each solution.
        """
        if solList is None:
            solList = self.solutionsList

        vec = [max(self.__normf(solT.objs)-self.__normf(sol.objs))
               for sol in solList]
        value = max(vec)

        return value, vec

    def __calcFirstW(
        self,
        goal: float = 1.0,
        eps: float = 0.00
    ) -> None:
        """Solve a linear optimization problem to compute the first weight
        vector (w) based on individual objective-optimal solutions.
        Assumes that the number of solutions is equal to the number of objectives.

        Args:
            goal (float): A target value for the objective. Default is 1.0.
            eps (float): Tolerance value for constraint relaxation. Default is 0.00.

        Raises:
            Exception: If the solver fails to find a feasible solution.
        """
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        assert self.M == Nsols, 'only fist W'

        # Create a gurobi model
        #prob = lp.LpProblem("max mean", lp.LpMaximize)
        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except:
            prob = mip.Model(sense=mip.MAXIMIZE)

        prob.verbose = 0

        # Creation of linear integer variables
        #w = list(lp.LpVariable.dicts('w', oidx, lowBound=0, upBound=1,
        #                             cat='Continuous').values())
        w = [prob.add_var(name='w', var_type=mip.CONTINUOUS, lb=0, ub=1) for i in oidx]
        
        uR = self.__globalL

        #v = lp.LpVariable('v', cat='Continuous')
        v = prob.add_var(name='v', var_type=mip.CONTINUOUS) 

        for conN, sols in enumerate(self.solutionsList):
            d, dvec = self.__calcD(sols)
            #expr = v-lp.lpDot(w, self.__normf(sols.objs))
            expr = v-mip.xsum(w[i]*self.__normf(sols.objs)[i] for i in oidx)
            prob += expr <= 0 #manter

        for i in oidx:
            prob += w[i] >= 0 #manter

        #prob += lp.lpSum([w[i] for i in oidx]) == 1
        prob += mip.xsum(w[i] for i in oidx) == 1
        #prob += v-lp.lpDot(w, self.__normf(uR))
        prob += v-mip.xsum(w[i]*self.__normf(uR)[i] for i in oidx)

        #try:
        #    grbs = lp.GUROBI(msg=False, OutputFlag=False)
        #    prob.solve(grbs)
        #except:
        #prob.solve()
            
        status = prob.optimize()

        #feasible = False if prob.status in [-1, -2] else True
        feasible = status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE

        if feasible:
            #w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
            #               for i in oidx])
            w_ = np.array([w[i].x if w[i].x >= 0 else 0 for i in oidx])
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            #fobj = lp.value(prob.objective)
            fobj = prob.objective_value
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            raise('Somethig wrong')

    def __calcW(self,
        goal: float = 1.0,
        eps: float = 0.00
    ) -> None:
        """ Solve a mixed-integer linear program (MILP) to compute an updated
        weighting vector (w) based on the current list of solutions. 

        Args:
            goal (float): A target value for the objective. Default is 1.0.
            eps (float): Constraint relaxation tolerance. Default is 0.00.

        Raises:
            Exception: If no feasible solution is found during optimization.
        """
        oidx = [i for i in range(self.M)]
        Nsols = len(self.solutionsList)
        # Create a gurobi model
        #prob = lp.LpProblem("max mean", lp.LpMaximize)
        try:
            prob = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.GRB) 
        except:
            prob = mip.Model(sense=mip.MAXIMIZE)
            
        prob.verbose = 0

        # Creation of linear integer variables
        w =  [ prob.add_var(name='w', lb=0, var_type=mip.CONTINUOUS) for i in oidx ]
        uR = [ prob.add_var(name='uR', lb=-np.inf, var_type=mip.CONTINUOUS) for i in oidx ]
        kp = [ prob.add_var(name='kp', lb=0, var_type=mip.CONTINUOUS) for i in range(Nsols) ]
        nu = [ prob.add_var(name='nu', lb=0, var_type=mip.CONTINUOUS) for i in oidx ]

        kpB = [ prob.add_var(name='kpB', var_type=mip.BINARY) for i in range(Nsols) ]
        nuB = [ prob.add_var(name='nuB', var_type=mip.BINARY) for i in oidx ]

        v = prob.add_var(name='v', lb=0, var_type=mip.CONTINUOUS)
        mu = prob.add_var(name='mu', lb=-np.inf, var_type=mip.CONTINUOUS)

        # Inherent constraints of this problem
        for value, sols in enumerate(self.solutionsList):
            expr = mip.xsum(self.__normw(sols.w)[i]*uR[i] for i in oidx)
            cons = self.__normw(sols.w) @ self.__normf(sols.objs)
            prob += expr >= cons*(1-eps)

        for i in oidx:
            expr = uR[i]-mip.xsum([kp[conN]*self.__normf(sols.objs)[i]
                                   for conN, sols in
                                   enumerate(self.solutionsList)])-nu[i]+mu
            prob += expr == 0

        for i in oidx:
            prob += uR[i] >= self.__normf(self.__globalL)[i] #mantem

        bigC = max(self.__normf(self.__globalU))

        for conN, sols in enumerate(self.solutionsList):
            expr = mip.xsum(w[i]*self.__normf(sols.objs)[i] for i in oidx)-v
            prob += expr >= 0 #mantem
            prob += expr <= kpB[conN]*bigC #mantem
            prob += kp[conN] >= 0 #mantem
            prob += kp[conN] <= (1-kpB[conN]) #mantem

        prob += mip.xsum(w[i] for i in oidx) == 1
        prob += mip.xsum(kp[i] for i in range(Nsols)) == 1

        for i in oidx:
            prob += w[i] >= 0 #mantem
            prob += w[i] <= nuB[i] #mantem
            prob += nu[i] >= 0 #mantem
            prob += nu[i] <= (1-nuB[i])*2*bigC #mantem
        
        prob += mip.xsum([mu])

        # desigualdades válidas
        prob += mu <= v

        rnd = np.array(sorted([0] +
                              [np.random.rand() for i in range(self.M-1)]
                              + [1]))
        w_ini = np.array([rnd[i+1]-rnd[i] for i in range(self.__M)])
        w_ini = w_ini/w_ini.sum()
        prob.start = [(wi, wii) for wi, wii in zip(w, w_ini)]

        #grbs = lp.GUROBI(epgap=self.__mip_gap, SolutionLimit=1,
        #                         msg=False, OutputFlag=False, Threads=1)

        prob.max_gap = self.__mip_gap
        prob.threads = 1
        prob.max_solutions = MAXINT
        prob.max_seconds = self.__time_limit

        #if self.__goal != float('inf'):
            #grbs = lp.GUROBI(timeLimit=self.__time_limit,
            #                         epgap=self.__mip_gap,
            #                         SolutionLimit=MAXINT,
            #                         msg=False, BestObjStop=self.__goal,
            #                         OutputFlag=False, Threads=1)


        #else:
            #grbs = lp.GUROBI(timeLimit=self.__time_limit,
            #                         epgap=self.__mip_gap,
            #                         SolutionLimit=MAXINT,
            #                         msg=False, OutputFlag=False,
            #                         Threads=1)

        status = prob.optimize()#max_solutions=None
            

        #feasible = False if prob.status in [-1, -2] else True
        feasible = status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE

        if feasible:
            #w_ = np.array([lp.value(w[i]) if lp.value(w[i]) >= 0 else 0
            #               for i in oidx])
            w_ = np.array([w[i].x if w[i].x >= 0 else 0 for i in oidx])
            w_ = w_/w_.sum()
            if self.__norm:
                w_ = w_/(self.__globalU-self.__globalL)
            #fobj = lp.value(prob.objective)
            fobj = prob.objective_value
            self.__w = np.array(w_)
            self.__importance = fobj
        else:
            raise('Non-feasible solution')


class monise():
    """MONISE: Many-Objective Non-Inferior Set Estimation.
    
    This algorithm incrementally constructs a Pareto frontier approximation by solving a
    sequence of weighted scalarization problems. It maintains and updates a candidate 
    list based on weight importance, improving the frontier coverage over time.
    """
    def __init__(
        self,
        weightedScalar: scalar,
        singleScalar: scalar,
        targetGap: float = 0.0,
        targetSize: int | None = None,
        redFact: float = float('inf'),
        smoothCount: int | None = None,
        nodeTimeLimit: float = float('inf'),
        nodeGap: float = 0.01,
        hotstart: list[scalar] = [],
        norm: bool = True
        ) -> None:
        """Initializes MONISE class

        Args:
            weightedScalar (scalar): An instance of a class solving the weighted scalarization
            singleScalar (scalar): An instance of a class solving single-objective problems.
            targetGap (float): Termination criterion based on importance ratio. Default is 0.0
            targetSize (int): Desired number of Pareto solutions.
                Defaults to 20 × number of objectives if not specified.
            redFact (float): Reduction factor used in adaptive refinement. Default is infinity.
            smoothCount (int): Number of smoothing iterations before stopping. Default is 0.
            nodeTimeLimit (float): Maximum time allowed (in seconds) per node optimization.
                Default is infinity.
            nodeGap (float): Acceptable optimization gap for each node. Default is 0.01.
            hotstart (list[scalar]): Initial solutions/models to warm-start the optimization.
            norm (bool): Whether to normalize objective vectors before comparison. Default is True.

        Raises:
            ValueError: If the provided `weightedScalar` or `singleScalar` does not implement the 
                expected scalarization interfaces.
        """
        self.__solutionsList = scalar_interface
        self.__solutionsList = w_interface
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
        self.__nodeTimeLimit = nodeTimeLimit
        self.__nodeGap = nodeGap
        self.__redFact = redFact
        self.__norm = norm
        if smoothCount is None:
            self.__smoothCount = 1 if nodeTimeLimit == float('inf') else 5
        else:
            self.__smoothCount = smoothCount

        self.__maxImp = 1
        self.__hotstart = hotstart
        self.__solutionsList = []
        self.__candidatesList = []

    def __del__(self) -> None:
        """
        Deletes the solutions list attribute from the object if it exists.
        
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

    @property
    def currImp(self) -> float:
        """Current importance score based on recent iterations.

        Returns:
            float: Maximum importance value from the last `smooth_count` iterations.
        """
        return max(self.__importances[-self.__smoothCount:])

    @property
    def maxImp(self) -> float:
        """Maximum importance value observed so far.

        Returns:
            float: The highest importance score recorded during optimization.
        """
        return self.__maxImp

    @property
    def importances(self) -> list[float]:
        """List of all importance values computed during optimization.

        Returns:
            list[float]: Historical record of importance scores.
        """
        return self.__importances

    def inicialization(self) -> weight_solv:
        """Initializes the optimization process.

        Finds the individual minima of each objective, computes the global lower 
        and upper bounds, and builds the first weighted solution. This sets up 
        the optimization for iterative refinement.

        Returns:
            weight_solv: The first weighted solution used to start the optimization.
        """
        self.__M = self.__singleScalar.M
        parents = []
        for i in range(self.__M):
            singleS = copy.copy(self.__singleScalar)
            logger.debug('Finding '+str(i+1)+'th individual minima')
            try:
                singleS.optimize(i, hotstart=self.hotstart)
            except:
                singleS.optimize(i)
            self.__solutionsList.append(singleS)
            parents.append(singleS)

        objsM = np.array([[o for o in p.objs] for p in parents])
        self.__globalL = objsM.min(0)
        self.__globalU = objsM.max(0)

        first_wsol = weight_solv(parents, self.__globalL, self.__globalU,
                                 self.__weightedScalar, norm=self.__norm)

        self.__maxImp = first_wsol.importance
        self.__importances = [first_wsol.importance]
        self.__goal = self.__maxImp

        return first_wsol

    def update(self, node: weight_solv, solution: scalar) -> None:
        """Updates the internal solution set with a new candidate.

        Args:
            node (weight_solv): The node that generated the solution.
            solution (scalar): New solution to be added.
        """
        self.solutionsList.append(solution)
        gap = self.currImp/self.__maxImp
        logger.debug(str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap))

    def _next(self) -> weight_solv:
        """Computes the next weighted solution.

        Returns:
            weight_solv: The next weighted solution to be optimized.
        """
        next_wsol = weight_solv(self.solutionsList, self.__globalL,
                                self.__globalU, self.__weightedScalar,
                                goal=self.currImp * self.__redFact,
                                time_limit=self.__nodeTimeLimit,
                                mip_gap=self.__nodeGap, norm=self.__norm)
        self.__importances += [next_wsol.importance]
        return next_wsol

    def optimize(self) -> None:
        """Runs the full MONISE optimization process.

        Initializes the algorithm, iteratively refines the solution set using
        weighted scalarization, selecting new weight vectors, solving the 
        scalarized problem, and updates the list of solutions until the stopping
        criteria are met.
        """
        start = time.perf_counter()
        next_wsol = self.inicialization()
        while (self.currImp / self.__maxImp > self.__targetGap and
               len(self.solutionsList) < self.__targetSize):
            solution = next_wsol.optimize(hotstart=self.hotstart)
            if next_wsol.best_solution_reached:
                logger.info("Best solution found.")
                break
            self.update(next_wsol, solution)
            next_wsol = self._next()

        self.__fit_runtime = time.perf_counter() - start
