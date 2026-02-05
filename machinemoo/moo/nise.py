import copy
import numpy as np
import numpy.typing as npt
from typing import List, Optional, cast
import heapq
import itertools

from machinemoo.moo.core import MOOptimizer
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface

__all__ = ["NISE"]

class WeightNode:
    """
    Represents a candidate region (interval/facet) for the NISE algorithm.
    
    It calculates a weight vector 'w' that is orthogonal to the hyperplane 
    defined by its parent solutions, and determines its importance based on 
    the potential error (distance) in that region.
    """
    def __init__(
        self,
        parents: List[w_interface],
        weighted_scalar: w_interface,
        solutions: List[w_interface] = [],
        distance_metric: str = 'l2',
    ) -> None:
        """
        Args:
            parents: List of parent solutions defining this region.
            global_lower: Global lower bounds (Utopia point).
            global_upper: Global upper bounds (Nadir point).
            weighted_scalar: Scalarization prototype.
            distance_metric: 'l2' (Euclidean) or 'algebraic' distance for importance.
            norm: Whether to normalize objectives.
        """
        self.parents = parents
        self.weighted_scalar = weighted_scalar
        self.solutions = solutions
        self.distance_metric = distance_metric
        self.M = int(weighted_scalar.M)
        
        self.w: npt.NDArray[np.float64]
        self.importance: float
        self._solution: Optional[w_interface] = None
        self.best_solution_reached = False

        # Compute state immediately
        self._calc_w()
        self._calc_importance()

    @property
    def useful(self) -> bool:
        """
        Checks if the current solution provides new information between parents.
        Used to prevent infinite refinement of the same point.
        """
        if self._solution is None:
            return False
            
        # Extract parent objectives matrix
        P = np.array([p.objs for p in self.parents])
        sol_objs = self._solution.objs
        
        # Check if solution is strictly bounded by parents (in the "middle")
        # Logic: It must be >= min(parents) AND <= max(parents) in all dims
        # AND it must not be equal to any parent.
        is_between = (
            np.all(sol_objs >= P.min(axis=0)) and 
            np.any(sol_objs <= P.max(axis=0))
        )
        
        is_duplicate = any(np.all(np.isclose(sol_objs, p_objs)) for p_objs in P)
        
        return bool(is_between and not is_duplicate)

    def optimize(self) -> w_interface:
        """Optimizes the scalarization using the calculated normal weight."""
        # Safety fallback
        weight = self.w if self.w is not None else np.ones(self.M) / self.M
        assert weight is not None
        
        # Warm start logic (simplificado)
        best_obj = self.w @ self.weighted_scalar.objs
        best_sol = self.weighted_scalar
        
        for s in self.solutions:
            # Produto escalar seguro
            val = self.w @ s.objs
            if val < best_obj:
                best_obj = val
                best_sol = s

        self._solution = copy.deepcopy(best_sol)
        self._solution.optimize(self.w)
        
        return self._solution

    def _calc_w(self) -> None:
        """
        Solves a linear system to find the weight vector 'w' such that the
        weighted sum is constant for all parent solutions (defining a hyperplane).
        """
        # System: w @ (p1 - p0) = 0, ..., w @ (pn - p0) = 0
        # Implemented as solving Xw = y where last row enforces sum(w)=1
        
        # Construct matrix X
        # Rows 0..M-1: Normalized objectives of parents
        # We append -1 to handle the hyperplane offset formulation w*x = c
        # X shape: (M+1, M+1)
        
        # Note: Original NISE logic for M=2 usually simplifies to slope calc.
        # General logic for M objectives requires M parents.
        
        # Prepare rows for linear system
        rows = []
        for p in self.parents:
            rows.append(list(p.objs) + [-1.0])
        
        # Add constraint: sum(w) = 1 (ignoring offset c)
        # Row format: [1, 1, ..., 1, 0]
        rows.append([1.0] * self.M + [0.0])
        
        X = np.array(rows)
        y = np.zeros(len(rows))
        y[-1] = 1.0 # Result for sum(w)=1 constraint

        # Solve
        try:
            res = np.linalg.solve(X, y)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            res, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        
        # The first M elements are the weights
        w_raw = res[:self.M]
        
        s = w_raw.sum()
        if s != 0:
            self.w = w_raw / s
        else:
            # find a random weight using the average of the parents
            self.w = np.mean([p.w for p in self.parents], axis=0)

    def _calc_importance(self, rcond: float=1e-10) -> None:
        """
        Calculates importance by finding the intersection of parents' hyperplanes.
        """
        if self.w is None:
            self._calc_w()

        X = [[i for i in p.w] for p in self.parents]
        y = [p.objs@p.w for p in self.parents]
        try:
            p = np.linalg.solve(X, y)
        except np.linalg.LinAlgError:
            p, residuals, rank, s = np.linalg.lstsq(X, y, rcond=rcond)

        # Find the reference point in w plane
        r = self.parents[0].objs #one of the parents


        if self.distance_metric == 'l2':
            self.importance = float((self.w@(r-p) /
                                    np.linalg.norm(self.w))**2)
        else:
            self.importance = float(self.w@(r-p))

class ExtremeNode(WeightNode):
    """
    Specialized WeightNode for extreme points (single parent).
    """
    def __init__(
        self,
        parent: w_interface,
        dimension: int,
        weighted_scalar: w_interface,
        solutions: List[w_interface] = [],
        distance_metric: str = 'l2',
    ) -> None:
        self.dimension = dimension
        super().__init__(
            parents=[parent],
            weighted_scalar=weighted_scalar,
            solutions=solutions,
            distance_metric=distance_metric,
        )

    @property
    def useful(self) -> bool:
        """
        Extreme nodes are always useful as they define boundaries.
        """
        return True
    
    def _calc_w(self) -> None:
        """
        For extreme nodes, set weight to focus entirely on the specific dimension.
        """
        w = np.zeros(self.M)
        w[self.dimension] = 1.0
        
        self.w = w

    def _calc_importance(self) -> None:
        parent = self.parents[0]

        assert parent is not None        
        assert self.w is not None
        
        p = parent.objs_lb
        r = parent.objs
        if self.distance_metric == 'l2':
            self.importance = float((self.w@(r-p) /
                               np.linalg.norm(self.w))**2)
        else:
            self.importance = float(self.w@(r-p))


class NISE(MOOptimizer):
    """
    Non-Inferior Set Estimation (NISE) algorithm.
    
    A classical method that iteratively explores the Pareto frontier by solving 
    weighted sum problems. Efficient for 2 objectives (bicriterion).
    """
    def __init__(
        self,
        weighted_scalar: w_interface,
        single_scalar: w_interface,
        # Base Parameters
        target_size: int = 50,
        time_limit: float = float('inf'),
        verbose: bool = False,
        debug: bool = False,
        # NISE Specific
        target_gap: float = 0.0,
        norm: bool = True,
        objective_metric: str = 'l2'
    ) -> None:
        super().__init__(
            target_size=target_size, 
            time_limit=time_limit, 
            verbose=verbose, 
            debug=debug
        )

        if not isinstance(weighted_scalar, (scalar_interface, w_interface)):
            raise ValueError("Scalarizers must implement correct interfaces.")

        self.weighted_scalar = weighted_scalar
        self.single_scalar = single_scalar
        self.target_gap = target_gap
        self.norm = norm
        self.objective_metric = objective_metric
        
        # State
        self.M: int = self.single_scalar.M
        self.global_lower: Optional[npt.NDArray[np.float64]] = None
        self.global_upper: Optional[npt.NDArray[np.float64]] = None
        self.max_imp: float = 1.0
        self.curr_imp: float = 1.0
        
        # Priority queue for candidates (sorted by importance)
        self.candidates_list: List[WeightNode] = []

    def initialize(self) -> None:
        """Initializes by finding individual minima and creating the first Node."""
        self.M = self.single_scalar.M
        if self.M != 2:
            self.logger.warning("NISE is theoretically optimized for 2 objectives. Behavior on M > 2 is experimental.")

        neig_o = []
        parents = []
        
        # 1. Find Individual Minima
        for i in range(self.M):
            self.logger.debug(f"Finding {i+1}th individual minimum")
            single_s = copy.copy(self.single_scalar)
            single_s.optimize(i)
            
            neig_o.append(single_s.objs)
            parents.append(single_s)
            solutions_list: List[w_interface] = cast(List[w_interface], self.solutions_list)
            extreme_node = ExtremeNode(single_s, i, self.weighted_scalar, solutions_list, self.objective_metric)
            self.update(single_s, extreme_node)

        # 2. Compute Bounds
        neig_o_arr = np.array(neig_o)
        self.global_lower = neig_o_arr.min(0)
        self.global_upper = neig_o_arr.max(0)
        assert self.global_lower is not None
        assert self.global_upper is not None

        # 3. Create First Node (The region covering the entire initial front)
        first_node = WeightNode(
            parents=parents,
            weighted_scalar=self.weighted_scalar,
            distance_metric=self.objective_metric
        )
        solution = first_node.optimize()
        self.update(solution, first_node)
        

    def select(self) -> Optional[WeightNode]:
        """
        Selects the next most important candidate node.
        NISE prioritizes regions with the largest potential error (importance).
        """
        # Filter out nodes with invalid/negative weights (bounded regions)
        # and pop the one with highest importance (list is sorted by importance)

        candidate: Optional[WeightNode] = None
        while self.candidates_list:
            candidate = self.candidates_list.pop()

            assert candidate.w is not None
            if  np.all(candidate.w >= 0):
                break
        return candidate

    def update(self, solution: w_interface, node: WeightNode) -> None:
        """
        Updates the solution list and branches the current node into sub-regions.
        """
        # 1. Base update
        super().update(solution, node)
        self._branch(node, solution)

    def _branch(self, node: WeightNode, solution: w_interface) -> None:
        """
        Splits the current node region into M new regions using the new solution.
        Updates the candidates list accordingly.
        """
        def branch_logic(node: WeightNode,
                         solution: w_interface) -> tuple[bool, bool, list[WeightNode]]:        
            assert node.w is not None
            child_nodes = []
            parents = node.parents
            
            if isinstance(node, ExtremeNode):
                parent = parents[0]
                if node.w@solution.objs >= node.w@parent.objs:
                    related = False
                    branch = False
                    return branch, related, child_nodes
                else:
                    child = ExtremeNode(
                        parent=solution,
                        dimension=node.dimension,
                        weighted_scalar=self.weighted_scalar,
                        distance_metric=self.objective_metric
                    )
                    # Aproveitamos o nó extremo existente
                    child_nodes.append(child)
                    return True, True, child_nodes

            if node.w@solution.objs >= node.w@parents[0].objs:
                related = False
                branch = False
                return branch, related, child_nodes
            else:
                related = True
                branch = True


            for p in parents:
                if p.w@solution.objs < p.w@p.objs:
                    branch = False
                    continue
            
                new_parents = [solution, p]
                
                child_node = WeightNode(
                    parents=new_parents,
                    weighted_scalar=self.weighted_scalar,
                    distance_metric=self.objective_metric
                )
            
                child_nodes.append(child_node)

            return branch, related, child_nodes
        branch, related, child_nodes = branch_logic(node, solution)
        
        if not branch:
            keep_mask = [True]*len(self.candidates_list)
            for i in range(len(self.candidates_list)-1, -1, -1):
                candidate = self.candidates_list[i]
                branch_, related_, child_nodes_ = branch_logic(candidate, solution)
                keep_mask[i] = not related_
                if not branch_:
                    child_nodes.extend(child_nodes_)
        else:
            keep_mask = [True]*len(self.candidates_list)

        #If main node is not related, add it back using child nodes
        if not related:
            child_nodes.append(node)
            
        self.candidates_list = merge_lists(self.candidates_list, keep_mask, child_nodes)
        if len(self.candidates_list) <= 2:
            self.initialize()

def merge_lists(large_sorted: list[WeightNode], keep_mask: list[bool], small_unsorted: list[WeightNode]) -> list[WeightNode]:
    """
    Merges a large sorted list with a small unsorted list in O(N + M) time.
    
    Args:
        large_sorted: The existing large list (must be already sorted).
        small_unsorted: The new small list to insert.
        key: A function to extract the comparison key (e.g., lambda x: x.importance).
    """
    def key(node: WeightNode) -> float:
        return node.importance
    # 1. Sort the small list. 
    # Cost: O(M log M) - Negligible since M is small.
    small_unsorted.sort(key=key)
    
    # 2. Criar um iterador filtrado da lista grande (Custo: O(1) para criar)
    # itertools.compress não aloca lista nova. Ele apenas avança o ponteiro 
    # se keep_mask[i] for True.
    large_filtered_iter = itertools.compress(large_sorted, keep_mask)

    # 2. Linear Merge.
    # heapq.merge is implemented in C. It iterates both lists once.
    # list() consumes the iterator to create the new structure.
    # Cost: O(N + M)
    return list(heapq.merge(large_filtered_iter, small_unsorted, key=key))