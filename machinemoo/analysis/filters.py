import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd

def filter_dominated(
    objectives: npt.NDArray[np.float64], 
    models: Optional[List[Any]] = None
) -> Tuple[npt.NDArray[np.float64], Optional[List[Any]]]:
    """
    Filters out dominated solutions from a set of objectives.
    
    A solution A is dominated by B if B is equal to or better than A in all 
    objectives (minimization), and strictly better in at least one.
    
    This function handles duplicates by keeping only the first occurrence.

    Args:
        objectives (np.ndarray): Array of shape (n_solutions, n_objectives).
        models (list, optional): List of models corresponding to the objectives.

    Returns:
        tuple:
            - np.ndarray: Filtered non-dominated objectives.
            - list or None: Corresponding filtered models (if provided).
    """
    if len(objectives) == 0:
        return objectives, models

    # 1. Identify non-dominated points
    is_efficient = np.ones(objectives.shape[0], dtype=bool)
    
    for i, c in enumerate(objectives):
        if is_efficient[i]:
            # Mark as inefficient if any OTHER point dominates c
            # c is dominated by o if: o <= c AND o != c
            # (We use a slightly more robust check handling duplicates implicitly by order)
            
            # Check strictly better or equal (dominance)
            # We want to find if there exists 'o' such that 'o' dominates 'c'
            
            # Using broadcasting to check against all currently efficient points
            mask = is_efficient.copy()
            mask[i] = False # Don't compare with self yet
            
            if np.any(np.all(objectives[mask] <= c, axis=1) & 
                      np.any(objectives[mask] < c, axis=1)):
                is_efficient[i] = False
                continue
                
            # If c is efficient, it might dominate others that haven't been processed yet
            # Remove points dominated BY c
            # (o >= c) and (o != c)
            dominated_by_c = np.all(objectives >= c, axis=1) & np.any(objectives > c, axis=1)
            is_efficient[dominated_by_c] = False
            
            # Handle duplicates: if strictly equal, keep only the current one (i), mark others False
            duplicates = np.all(objectives == c, axis=1)
            duplicates[i] = False # Keep self
            is_efficient[duplicates] = False

    # 2. Filter results
    filtered_objs = objectives[is_efficient]
    filtered_models = None
    if models is not None:
        filtered_models = [models[i] for i, valid in enumerate(is_efficient) if valid]

    return filtered_objs, filtered_models