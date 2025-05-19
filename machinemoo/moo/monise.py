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
import numpy as np
import copy
import logging
import time
import mip

from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "monise"
]


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

MAXINT = 200000000000000

# -*- coding: utf-8 -*-
"""
Many Objective Noninferior Estimation utils

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

import copy
import mip
import numpy as np

MAXINT = 2000000000

class weight_solv():
    def __init__(self, solutionsList, globalL, globalU, weightedScalar,
                 goal=float('inf'), time_limit=10, mip_gap=0.01, norm=False):
        self.__weightedScalar = weightedScalar
        self.__M = solutionsList[0].M
        self.__globalL, self.__globalU = globalL, globalU
        self.solutionsList = solutionsList
        self.__goal = goal
        self.__time_limit = time_limit
        self.__mip_gap = mip_gap
        self.__norm = norm
        if len(self.solutionsList) == self.M:
            self.__calcFirstW()
        else:
            self.__calcW(goal=goal)

    @property
    def M(self):
        return self.__M

    @property
    def importance(self):
        return self.__importance

    #@property
    #def parents(self):
    #    return self.__parents

    @property
    def solution(self):
        return self.__solution

    @property
    def w(self):
        return self.__w

    def optimize(self, hotstart=None):
        self.__solution = copy.copy(self.__weightedScalar)
        try:
            self.__solution.optimize(self.w, [])
        except:
            self.__solution.optimize(self.w)
        return self.__solution

    def __normf(self, obj):
        if self.__norm:
            return (obj-self.__globalL)/(self.__globalU-self.__globalL)
        else:
            return (obj-self.__globalL)

    def __normw(self, w):
        if self.__norm:
            w_ = w*(self.__globalU-self.__globalL)
            return w_/w_.sum()
        else:
            return w

    def __calcD(self, solT, solList=None):

        if solList is None:
            solList = self.solutionsList

        vec = [max(self.__normf(solT.objs)-self.__normf(sol.objs))
               for sol in solList]
        value = max(vec)

        return value, vec

    def __calcFirstW(self, goal=1, eps=0.00):
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

    def __calcW(self, goal=1, eps=0.00):
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
    def __init__(self, weightedScalar, singleScalar, targetGap=0.0,
                 targetSize=None, redFact=float('inf'), smoothCount=None,
                 nodeTimeLimit=float('inf'), nodeGap=0.01, hotstart=[],
                 norm=True):
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

    def __del__(self):
        if hasattr(self, '__solutionsList'):
            del self.__solutionsList

    @property
    def targetSize(self): return self.__targetSize

    @property
    def targetGap(self): return self.__targetGap

    @property
    def solutionsList(self): return self.__solutionsList

    @property
    def hotstart(self): return self.__hotstart+self.solutionsList

    @property
    def currImp(self):
        return max(self.__importances[-self.__smoothCount:])

    @property
    def maxImp(self): return self.__maxImp

    @property
    def importances(self): return self.__importances

    def inicialization(self):
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

    def update(self, node, solution):
        self.solutionsList.append(solution)
        gap = self.currImp/self.__maxImp
        logger.debug(str(len(self.solutionsList))+'th solution' +
                     ' - importance: ' + str(gap))

    def _next(self):
        next_wsol = weight_solv(self.solutionsList, self.__globalL,
                                self.__globalU, self.__weightedScalar,
                                goal=self.currImp * self.__redFact,
                                time_limit=self.__nodeTimeLimit,
                                mip_gap=self.__nodeGap, norm=self.__norm)
        self.__importances += [next_wsol.importance]
        return next_wsol

    def optimize(self):
        start = time.perf_counter()
        next_wsol = self.inicialization()
        while (self.currImp / self.__maxImp > self.__targetGap and
               len(self.solutionsList) < self.__targetSize):
            solution = next_wsol.optimize(hotstart=self.hotstart)
            self.update(next_wsol, solution)
            next_wsol = self._next()

        self.__fit_runtime = time.perf_counter() - start
