import copy
import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, cast, Tuple
import pyomo.environ as pyo
from pyomo.contrib import appsi

from machinemoo.moo.core import MOOptimizer, IPSolvableMixin
from machinemoo.utils.typing import scalar
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = ["MONISE"]

class WeightNode:
    """
    Resolve o problema de otimização de pesos do MONISE usando Pyomo.
    """
    def __init__(
        self,
        solutions: List[scalar],
        global_lower: npt.NDArray[np.float64],
        global_upper: npt.NDArray[np.float64],
        weighted_scalar: scalar,
        time_limit: float = 10.0,
        mip_gap: float = 0.01,
        norm: bool = True
    ) -> None:
        self.solutions = solutions
        self.M = solutions[0].M
        self.global_lower = global_lower
        self.global_upper = global_upper
        self.weighted_scalar = weighted_scalar
        self.norm = norm
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        
        self.w: Optional[npt.NDArray[np.float64]] = None
        self.importance: float = 0.0
        self.best_solution_reached = False
        self._solution: Optional[scalar] = None

        # Decisão de qual solver usar baseada no estado atual
        if len(self.solutions) == self.M:
            self.w, self.importance = self._solve_linear_problem()
        else:
            self.w, self.importance = self._solve_milp_problem()

    def optimize(self) -> scalar:
        """Otimiza a escalarização com o peso calculado."""
        # Fallback de segurança se o solver falhou silenciosamente
        assert self.w is not None, "Weight vector 'w' was not computed."

        # Warm start logic (simplificado)
        best_obj = self.w @ self.weighted_scalar.objs
        best_sol = self.weighted_scalar
        
        for s in self.solutions:
            # Produto escalar seguro
            val = self.w @ s.objs
            if val < best_obj:
                best_obj = val
                best_sol = s

        self._solution = copy.copy(best_sol)
        self._solution.optimize(self.w)
        
        if best_sol is not None and np.allclose(self._solution.objs, best_sol.objs):
            self.best_solution_reached = True
            
        return self._solution

    def _solve_linear_problem(self) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Encapsula a resolução do problema linear inicial (Primeiro W).
        Retorna (weights, importance).
        """
        # Dados locais para evitar self dentro do modelo
        M_idx = range(self.M)
        # Extrair dados para estruturas simples para evitar overhead
        s_objs = [s.objs_lb for s in self.solutions]
        g_lower = self.global_lower

        m = pyo.ConcreteModel()
        
        # Variáveis
        # Usamos type: ignore aqui pois Pyomo adiciona atributos dinamicamente
        m.w = pyo.Var(M_idx, domain=pyo.NonNegativeReals, bounds=(0, 1)) # type: ignore
        m.v = pyo.Var(domain=pyo.Reals) # type: ignore
        
        # Restrições
        m.cons = pyo.ConstraintList() # type: ignore
        for obj_vec in s_objs:
            # v <= w @ objs
            lhs = m.v - sum(m.w[i] * obj_vec[i] for i in M_idx) # type: ignore
            m.cons.add(lhs <= 0) # type: ignore

        # Simplex: sum(w) == 1
        m.simplex = pyo.Constraint(expr=sum(m.w[i] for i in M_idx) == 1) # type: ignore
        
        # Objetivo: Maximizar v - w @ global_lower
        def obj_rule(model: Any) -> Any:
            return model.v - sum(model.w[i] * g_lower[i] for i in M_idx)
        
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize) # type: ignore

        return self._run_solver(m) # type: ignore

    def _solve_milp_problem(self, eps: float = 0.0) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Encapsula a resolução do problema MILP (Próximos W).
        Retorna (weights, importance).
        """
        # Dados locais
        M_idx = range(self.M)
        K_idx = range(len(self.solutions))
        
        s_objs = [s.objs_lb for s in self.solutions]
        # Se 'w' não existir na solução anterior, usa uniforme
        s_ws = [getattr(s, 'w', np.ones(self.M)/self.M) for s in self.solutions]
        
        g_lower = self.global_lower
        g_upper = self.global_upper
        
        # Big-M
        big_c = np.max(g_upper) if g_upper is not None else 100.0

        m = pyo.ConcreteModel()

        # Variáveis (com type: ignore para silenciar o Pylance sobre atributos dinâmicos)
        m.w = pyo.Var(M_idx, domain=pyo.NonNegativeReals) # type: ignore
        m.uR = pyo.Var(M_idx, domain=pyo.Reals) # type: ignore
        m.kp = pyo.Var(K_idx, domain=pyo.NonNegativeReals, bounds=(0, 1)) # type: ignore
        m.kpB = pyo.Var(K_idx, domain=pyo.Binary) # type: ignore
        
        m.nu = pyo.Var(M_idx, domain=pyo.NonNegativeReals) # type: ignore
        m.nuB = pyo.Var(M_idx, domain=pyo.Binary) # type: ignore
        m.v = pyo.Var(domain=pyo.Reals) # type: ignore
        m.mu = pyo.Var(domain=pyo.Reals) # type: ignore

        m.cons = pyo.ConstraintList() # type: ignore

        # 1. Restrições no uR (Hiperplanos das soluções existentes)
        for k in K_idx:
            lhs = sum(s_ws[k][i] * m.uR[i] for i in M_idx) # type: ignore
            rhs = np.dot(s_ws[k], s_objs[k]) * (1 - eps)
            m.cons.add(lhs >= rhs) # type: ignore

        # 2. Definição de uR baseada na região ativa (k)
        for i in M_idx:
            term_k = sum(m.kp[k] * s_objs[k][i] for k in K_idx) # type: ignore
            m.cons.add(m.uR[i] - term_k - m.nu[i] + m.mu == 0) # type: ignore

        # 3. uR >= Utopia
        for i in M_idx:
            m.cons.add(m.uR[i] >= g_lower[i]) # type: ignore

        # 4. Seleção da região ativa
        for k in K_idx:
            val_k = sum(m.w[i] * s_objs[k][i] for i in M_idx) # type: ignore
            m.cons.add(val_k - m.v >= 0) # type: ignore
            m.cons.add(val_k - m.v <= m.kpB[k] * big_c) # type: ignore
            m.cons.add(m.kp[k] + m.kpB[k] <= 1) # type: ignore

        # Simplex constraints
        m.cons.add(sum(m.w[i] for i in M_idx) == 1) # type: ignore
        m.cons.add(sum(m.kp[k] for k in K_idx) == 1) # type: ignore

        # 5. Restrições Duais
        for i in M_idx:
            m.cons.add(m.w[i] <= m.nuB[i]) # type: ignore
            m.cons.add(m.nu[i] + 2 * big_c * m.nuB[i] <= 2 * big_c) # type: ignore

        m.cons.add(m.mu <= m.v) # type: ignore

        # Objetivo
        m.obj = pyo.Objective(expr=m.mu, sense=pyo.maximize) # type: ignore

        return self._run_solver(m) # type: ignore

    def _run_solver(self, model: pyo.ConcreteModel) -> Tuple[npt.NDArray[np.float64], float]:
        """
        Executa o solver e extrai os resultados de forma segura.
        Aqui centralizamos a interação com o appsi/gurobi.
        """
        solver = appsi.solvers.Gurobi()
        solver.config.time_limit = self.time_limit
        solver.config.mip_gap = self.mip_gap
        
        # Padrões de retorno em caso de falha
        default_w = np.ones(self.M) / self.M
        default_imp = 0.0

        try:
            res = solver.solve(model)
            # Verifica condição de terminação (optimal ou feasible)
            if res.termination_condition not in [appsi.base.TerminationCondition.optimal, appsi.base.TerminationCondition.maxTimeLimit]:
                 return default_w, default_imp

            # Extração segura usando list comprehension
            # O cast para Any/float evita erros de tipo no retorno do pyo.value
            w_vals = [pyo.value(model.w[i]) for i in range(self.M)] # type: ignore
            w_arr = np.array(w_vals, dtype=np.float64)
            
            # Normalização numérica
            if w_arr.sum() > 1e-9:
                w_arr /= w_arr.sum()
            
            # Tenta pegar o valor objetivo, fallback se falhar
            try:
                imp = float(pyo.value(model.obj)) # type: ignore
            except Exception:
                imp = 0.0
                
            return w_arr, imp

        except Exception:
            # Em produção, você pode querer logar o erro 'e'
            return default_w, default_imp


class MONISE(IPSolvableMixin, MOOptimizer):
    """
    MONISE: Many-Objective Non-Inferior Set Estimation.
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar,
        # Base Parameters (MOOPTimizer))
        target_size: int = 20,
        time_limit: float = float('inf'),   
        verbose: bool = False,
        debug: bool = False,
        # IPSolvableMixin Parameters
        node_time_limit: float = float('inf'),
        node_gap: float = 0.01,
    ) -> None:
        super().__init__(
            target_size=target_size,
            time_limit=time_limit,
            verbose=verbose,
            debug=debug,
            node_time_limit=node_time_limit,
            node_gap=node_gap,
        )

        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, single_interface))):
            raise ValueError("Scalarizers must implement correct interfaces.")

        # Setting Scalarizers
        self.weighted_scalar = weighted_scalar
        self.single_scalar = single_scalar
        
        # Setting State
        self.M: int = 0
        self.global_lower: Optional[npt.NDArray[np.float64]] = None
        self.global_upper: Optional[npt.NDArray[np.float64]] = None
        self.importances: List[float] = []
        self._next_node: Optional[WeightNode] = None

    def initialize(self) -> None:
        self.M = self.single_scalar.M
        
        # 1. Encontrar Mínimos Individuais
        for i in range(self.M):
            single_s = copy.copy(self.single_scalar)
            self.logger.debug(f"Finding {i+1}th individual minima")
            single_s.optimize(i)
            self.update(single_s, None)

        # 2. Calcular Limites Globais
        objs_lb_matrix = np.array([s.objs_lb for s in self.history_list])
        self.global_lower = objs_lb_matrix.min(axis=0)
        
        objs_matrix = np.array([s.objs for s in self.history_list])
        self.global_upper = objs_matrix.max(axis=0)

        # 3. Criar Primeiro Nó (Otimização Linear Inicial)
        first_node = WeightNode(
            solutions=self.solutions_list,
            global_lower=cast(npt.NDArray[np.float64], self.global_lower),
            global_upper=cast(npt.NDArray[np.float64], self.global_upper),
            weighted_scalar=self.weighted_scalar,
        )
        
        self.importances = [first_node.importance]
        self._next_node = first_node

    def select(self) -> Optional[WeightNode]:
        current_node = self._next_node
        
        # Prepara o próximo nó (Lookahead do algoritmo MONISE)
        if current_node is not None:
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

    def update(self, solution: scalar, node: Any) -> None:
        super().update(solution, node)
        if self.global_lower is not None:
            self.global_lower = np.minimum(self.global_lower, solution.objs_lb)