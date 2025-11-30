# -*- coding: utf-8 -*-
"""
Rennen: A posteriori multiobjective optimization method.

Based on polyhedral approximation (Sandwich Algorithm) and dummy points.

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas

Reference:
    Enhancement of Sandwich Algorithms for Approximating Higher-Dimensional
    Convex Pareto Sets
    Gijs Rennen, Edwin R. van Dam, and Dick den Hertog
    INFORMS Journal on Computing 2011 23:4, 493-517
"""

import copy
import numpy as np
import numpy.typing as npt
import pulp as lp
from scipy.spatial import ConvexHull
from typing import Any, List, Optional, Set, Dict, Tuple, cast

from machinemoo import get_logger
from machinemoo.moo.core import MOOptimizer
from machinemoo.utils.typing import scalar
from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface

__all__ = [
    "Rennen"
]

logger = get_logger(f"moo.{__name__}")

# --- Helper Functions ---

def convex_combination(points: List[npt.NDArray[np.float64]], point: npt.NDArray[np.float64]) -> bool:
    """
    Calculates if a point is a convex combination of other points in the hull.
    Used to avoid adding unnecessary points to the hull calculation.
    """
    n_points = len(points)
    n_dim = len(point)
    points_idx = list(range(n_points))
    dim_idx = list(range(n_dim))

    # Create LP problem
    prob = lp.LpProblem("max_mean", lp.LpMinimize)

    # Variables: alpha coefficients for convex combination
    alpha = lp.LpVariable.dicts('alpha', points_idx, lowBound=0, cat='Continuous')

    # Constraints: sum(alpha * points) == point
    for j in dim_idx:
        prob += (lp.lpSum([alpha[i] * points[i][j] for i in points_idx]) == point[j])

    # Constraint: sum(alpha) == 1 (Convex combination)
    prob += lp.lpSum([alpha[i] for i in points_idx]) == 1

    # Special handling for exact duplicates to prevent trivial solutions
    # If point matches points[i], force alpha[i] = 0 to check if it can be formed by OTHERS
    # (Though logic in original code seems to check if it is *strictly* contained)
    for i in points_idx:
        if np.all(np.isclose(points[i], point)):
            prob += alpha[i] == 0

    # Solve
    # Suppress output
    solver = lp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Feasible means it IS a convex combination
    feasible = (prob.status not in [lp.LpStatusInfeasible, lp.LpStatusUnbounded])
    return feasible


def already_found(points: List[npt.NDArray[np.float64]], point: npt.NDArray[np.float64]) -> bool:
    """Checks if a point already exists in the list."""
    return any(np.all(np.isclose(p, point)) for p in points)


class WeightNode:
    """
    Helper class to calculate weight vectors and solve scalarizations for Rennen.
    Represents a facet/candidate in the sandwich approximation.
    """
    def __init__(
        self,
        w: npt.NDArray[np.float64],
        solutions_list: List[scalar],
        global_lower: npt.NDArray[np.float64],
        global_upper: npt.NDArray[np.float64],
        weighted_scalar: scalar,
        point: npt.NDArray[np.float64],
        norm: bool = True
    ) -> None:
        self.weighted_scalar = weighted_scalar
        self.M = weighted_scalar.M
        self.global_lower = global_lower
        self.global_upper = global_upper
        self.norm = norm
        self.target_point = point
        
        self.w: npt.NDArray[np.float64] = np.zeros(self.M)
        self.importance: float = 0.0
        self.uR: Optional[npt.NDArray[np.float64]] = None
        self._solution: Optional[scalar] = None
        self.best_solution_reached = False

        self._calc_w(w)
        self.calc_importance(solutions_list)

    def optimize(self) -> scalar:
        """Runs the optimization for this weight vector."""
        self._solution = copy.copy(self.weighted_scalar)
        self._solution.optimize(self.w)
        return self._solution

    def _calc_w(self, w: npt.NDArray[np.float64]) -> None:
        """Processes and normalizes the weight vector."""
        # Adjust weight direction if necessary (gradient direction)
        # Note: Logic preserved from original
        norm_gl = self.global_lower
        norm_gu = self.global_upper
        
        if (w @ norm_gl) < (w @ norm_gu):
            w = w / np.abs(w).sum()
        else:
            w = w / (-np.abs(w).sum())

        if self.norm:
            diff = self.global_upper - self.global_lower
            diff = np.where(diff == 0, 1.0, diff)
            w = w / diff

        self.w = np.abs(w)

    def calc_importance(self, solutions_list: List[scalar]) -> None:
        """
        Calculates the importance (error bound) of this facet.
        Uses cached uR if valid, otherwise solves LP.
        """
        # Check if cached uR is still valid
        if self.uR is not None:            
            # Check feasibility against all solutions
            is_valid = True
            for sol in solutions_list:
                # Cast to runtime attributes to satisfy the type checker
                norm_w_sol = cast(npt.NDArray[np.float64], getattr(sol, 'w'))
                norm_obj = cast(npt.NDArray[np.float64], getattr(sol, 'objs'))
                
                # Check intersection of half-spaces
                if not (norm_w_sol @ self.uR >= norm_obj @ norm_w_sol):
                    is_valid = False
                    break
            
            if is_valid:
                return

        self._solve_importance_lp(solutions_list)

    def _solve_importance_lp(self, solutions_list: List[scalar]) -> None:
        """
        Solves LP to find the lower bound point (uR) and importance.
        """
        oidx = list(range(self.M))
        prob = lp.LpProblem("Lower_point", lp.LpMinimize)

        # Variable uR (Lower bound point estimate)
        uR = lp.LpVariable.dicts('uR', oidx, cat='Continuous')

        for sol in solutions_list:
            # Cast to runtime attributes to satisfy the type checker
            norm_w_sol = cast(npt.NDArray[np.float64], getattr(sol, 'w'))
            norm_obj = cast(npt.NDArray[np.float64], getattr(sol, 'objs'))
            
            expr = lp.lpSum([norm_w_sol[i] * uR[i] for i in oidx])
            cons = norm_obj @ norm_w_sol
            prob += (expr >= cons)
            prob += (expr >= cons)

        # Objective: Minimize projection of uR onto current weight direction
        norm_w_curr = self.w
        prob += lp.lpSum([norm_w_curr[i] * uR[i] for i in oidx])

        # Solve
        solver = lp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        feasible = (prob.status not in [lp.LpStatusInfeasible, lp.LpStatusUnbounded])

        if feasible:
            self.uR = np.array([lp.value(uR[i]) for i in oidx])
        else:
            raise RuntimeError("Non-feasible solution in Importance calculation")


class Rennen(MOOptimizer):
    """
    Rennen: A posteriori MOO method based on polyhedral approximation (Sandwich Algorithm).
    """
    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar,
        target_size: int = 50,
        norm: bool = True,
        time_limit: float = float('inf'),
        verbose: bool = False,
        debug: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(target_size=target_size, time_limit=time_limit, verbose=verbose, debug=debug)

        if (not isinstance(weighted_scalar, (scalar_interface, w_interface)) or
            not isinstance(single_scalar, (scalar_interface, single_interface))):
            raise ValueError("Scalarizers must implement correct interfaces.")

        self.weighted_scalar = weighted_scalar
        self.single_scalar = single_scalar
        self.norm = norm
        
        self.M: int = 0
        self.global_lower: Optional[npt.NDArray[np.float64]] = None
        self.global_upper: Optional[npt.NDArray[np.float64]] = None
        self.max_u: Optional[npt.NDArray[np.float64]] = None
        
        self.candidates_list: Dict[Tuple[int, ...], WeightNode] = {}
        self.selected_simplices: List[Set[int]] = []
        
        self.hull_points: List[npt.NDArray[np.float64]] = []
        self.convex_hull: Optional[ConvexHull] = None
        self._next_facet: Optional[Set[int]] = None
        
        # Tracking
        self.importances: List[float] = []

    def initialize(self) -> None:
        self.M = self.single_scalar.M
        neig_o = []
        parents = []

        # 1. Find Individual Minima
        for i in range(self.M):
            single_s = copy.copy(self.single_scalar)
            self.logger.debug(f"Finding {i+1}th individual minima")
            single_s.optimize(i)
            
            neig_o.append(single_s.objs)
            parents.append(single_s)
            self.update(None, single_s)

        # 2. Compute Bounds
        neig_o_arr = np.array(neig_o)
        self.global_lower = neig_o_arr.min(0)
        self.global_upper = neig_o_arr.max(0)
        self.max_u = neig_o_arr.max(0)

        # 3. Create Initial Node (Average weight)
        initial_w = np.ones(self.M) / self.M
        
        # Note: Rennen starts by optimizing a central weight formed by the extreme points
        first_node = WeightNode(
            w=initial_w,
            solutions_list=parents,
            global_lower=cast(npt.NDArray[np.float64], self.global_lower),
            global_upper=cast(npt.NDArray[np.float64], self.global_upper),
            weighted_scalar=self.weighted_scalar,
            point=parents[0].objs, # Reference point, usually not critical for first iter
            norm=self.norm
        )
        
        # Prepare for first selection (manually triggering optimization of the central point)
        # We simulate this as a "candidate" to be picked up by select()
        # For the very first step, we usually just run it. 
        # But to fit BaseMOO loop, we need to queue it.
        # Actually, Rennen logic typically: Init -> Optimize Central -> Then Branch.
        # We can optimize the central point right here in initialize and branch.
        
        self.logger.debug("Optimizing initial central weight...")
        central_sol = first_node.optimize()
        self.update(first_node, central_sol)
        
        # Branching is handled inside update -> _branch

    def select(self) -> Optional[WeightNode]:
        """
        Selects the next most important facet/candidate from the list.
        """
        if self._next_facet is None:
            return None
            
        facet_key = tuple(self._next_facet)
        if facet_key in self.candidates_list:
            next_node = self.candidates_list[facet_key]
            self.importances.append(next_node.importance)
            self.selected_simplices.append(self._next_facet)
            return next_node
            
        return None

    def update(self, node: Any, solution: scalar) -> None:
        """
        Updates solution set, bounds, and the convex hull approximation (branching).
        """
        # 1. BaseMOO update (filtering)
        super().update(node, solution)
        
        # 2. Update Max Upper Bound (used for dummy points)
        if self.max_u is not None:
            self.max_u = np.maximum(self.max_u, solution.objs)
            
        # 3. Update Convex Hull and Candidates
        self._branch(node, solution)
        
        gap = self._next_node_importance() if self._next_facet else 0.0
        self.logger.debug(f"Solution added. Next max importance: {gap}")

    def _next_node_importance(self) -> float:
        if self._next_facet:
            return self.candidates_list[tuple(self._next_facet)].importance
        return 0.0

    def _dummy_points(self, solution: scalar) -> List[npt.NDArray[np.float64]]:
        """Generates dummy points for convex hull construction (Definition 7 in paper)."""
        # Ensure global bounds are set
        assert self.global_lower is not None and self.global_upper is not None, "Global bounds must be initialized before generating dummy points"
        
        points = []
        # Use normalized solution and normalized upper bound to build consistent dummy points
        norm_sol = solution.objs
        norm_u = self.global_upper
        
        for i in range(self.M):
            point = norm_sol.copy()
            # Push point far out along axis i
            point[i] = norm_u[i] * self.M + norm_u[i] * 0.01
            points.append(point)
        return points

    def _new_points(self, solution: scalar) -> List[npt.NDArray[np.float64]]:
        norm_sol = solution.objs
        return [norm_sol] + self._dummy_points(solution)

    def _is_dummy(self, point: npt.NDArray[np.float64]) -> bool:
        # Ensure global_upper is initialized before performing arithmetic
        if self.global_upper is None:
            raise AssertionError("Global upper bounds must be initialized before calling _is_dummy")
        norm_u = self.global_upper
        # np.any returns a numpy.bool_, convert explicitly to Python bool
        return bool(np.any(point > norm_u * self.M))

    def _all_dummy(self, points: List[npt.NDArray[np.float64]]) -> bool:
        return all(self._is_dummy(p) for p in points)

    def _sel_not_dummy(self, points: List[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        for point in points:
            if not self._is_dummy(point):
                return point
        return points[0] # Fallback

    def _branch(self, node: Any, solution: scalar) -> None:
        """
        Updates the set of candidates by re-computing the Convex Hull with the new solution.
        """
        # Ensure bounds are initialized before creating WeightNode instances
        assert self.global_lower is not None and self.global_upper is not None, "Global bounds must be initialized before branching"
        old_candidates = self.candidates_list
        self.candidates_list = {}

        # 1. Update Hull Points
        # Add new points (solution + dummies) to the hull set
        if self.convex_hull is None:
            # First time: gather all current solutions
            all_points = []
            for sol in self.solutions_list:
                all_points.extend(self._new_points(sol))
            
            # Filter redundant points
            self.hull_points = [p for p in all_points if not convex_combination(all_points, p)]
            self.convex_hull = ConvexHull(self.hull_points, qhull_options='Q12')
            
        elif not already_found(self.hull_points, solution.objs):
            # Incremental update
            new_pts = self._new_points(solution)
            # Add only if not convex combination of existing
            # Note: Checking against current hull points + new points
            current_plus_new = self.hull_points + new_pts
            
            valid_new_pts = [p for p in new_pts if not convex_combination(current_plus_new, p)]
            self.hull_points += valid_new_pts
            
            if valid_new_pts:
                self.convex_hull = ConvexHull(self.hull_points, qhull_options='Q12')

        # 2. Iterate over Facets (Simplices)
        n_facets = self.convex_hull.simplices.shape[0]
        next_facet = None
        next_importance = -float('inf')

        for i in range(n_facets):
            simplice_indices = self.convex_hull.simplices[i]
            simplice_set = set(simplice_indices)
            simplice_key = tuple(simplice_set)

            # Equation of the plane: w @ x + b = 0 -> w is normal vector
            # ConvexHull.equations returns [w0, w1, ..., wn, offset]
            eq = self.convex_hull.equations[i]
            w_ch = eq[:-1]

            points_on_facet = [self.convex_hull.points[s] for s in simplice_set]

            # Filter irrelevant facets
            # 1. All points are dummy -> Boundary facet facing infinity
            # 2. Normal vector has significant negative components (should be positive for Pareto)
            if self._all_dummy(points_on_facet):
                continue
            
            # Small tolerance for numerical noise in normal vector
            if np.any(w_ch < -1e-10) and not np.all(w_ch <= 0):
                continue

            # Skip already processed facets
            if simplice_set in self.selected_simplices:
                continue

            # Check if we already have a node for this facet from previous iteration
            if simplice_key in old_candidates:
                new_node = old_candidates[simplice_key]
                # Re-calculate importance because the 'solutions_list' (constraints) changed
                # Only re-calc if it might be the next best
                if new_node.importance >= next_importance:
                    new_node.calc_importance(self.solutions_list)
            else:
                # Create new node
                point = self._sel_not_dummy(points_on_facet)
                new_node = WeightNode(
                    w=w_ch,
                    solutions_list=self.solutions_list,
                    global_lower=cast(npt.NDArray[np.float64], self.global_lower),
                    global_upper=cast(npt.NDArray[np.float64], self.global_upper),
                    weighted_scalar=self.weighted_scalar,
                    point=point,
                    norm=self.norm
                )

            self.candidates_list[simplice_key] = new_node

            # Track best candidate
            if new_node.importance >= next_importance:
                next_importance = new_node.importance
                next_facet = simplice_set

        self._next_facet = next_facet