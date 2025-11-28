import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd
from pymcdm.methods import TOPSIS

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


def topsis_model_selection(
    losses: npt.NDArray[np.float64], 
    metrics: Optional[Dict[str, List[float]]] = None, 
    criteria_types: Optional[List[int]] = None, 
    weights: Optional[List[float]] = None, 
    model_names: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int, str, npt.NDArray[np.float64]]:
    """
    Perform TOPSIS to select the best model(s) from losses and optional metrics.

    Returns ranking DataFrame and best model info.
    """
    losses = np.array(losses)
    n_models = losses.shape[0]

    # Inicializar matriz
    data = losses

    # Concatena métricas opcionais
    if metrics:
        for metric_name, values in metrics.items():
            metric_array = np.array(values).reshape(-1, 1)
            data = np.hstack([data, metric_array])

    # Definir critérios
    if criteria_types is None:
        # losses = -1 (minimizar), métricas = 1 (maximizar, assumindo acurácia/score)
        criteria_types = [-1] * losses.shape[1] 
        assert criteria_types is not None
        if metrics:
            criteria_types += [1] * len(metrics)

    # Pesos
    if weights is None:
        n_criteria = data.shape[1]
        weights = np.ones(n_criteria) / n_criteria
        assert weights is not None

    # Nomes de modelos
    if model_names is None:
        model_names = [f"M{i}" for i in range(n_models)]

    # Aplica TOPSIS
    topsis = TOPSIS()
    scores = np.array(topsis(data, weights, criteria_types))

    # Ranking (ordenando)
    ranking_order = np.argsort(-scores)  # do maior score para o menor
    
    df = pd.DataFrame({
        "Model": [model_names[i] for i in ranking_order],
        "TOPSIS Score": scores[ranking_order],
        "Original Index": ranking_order
    })
    df["Rank"] = np.arange(1, n_models + 1)

    # Melhor modelo (primeiro do ranking)
    if n_models > 0:
        best_index = int(ranking_order[0])
        best_model_name = model_names[best_index]
        best_losses = losses[best_index]
    else:
        best_index = -1
        best_model_name = "None"
        best_losses = np.array([])

    return df, best_index, best_model_name, best_losses


def select_lowest_median_solution(
    objectives: npt.NDArray[np.float64],
    models: List[Any],
    objective_thresholds: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Returns the solution with the lowest median objective value, optionally filtering by thresholds.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must match.")

    filtered = []
    for i, (obj, model) in enumerate(zip(objectives, models)):
        # Skip purely zero solutions (often artifacts or errors)
        if np.any(np.isclose(obj, 0.0)):
            continue

        discard = False
        if objective_thresholds:
            for idx, thresh in objective_thresholds.items():
                val = float(np.array(obj)[idx])
                if val > thresh:
                    discard = True
                    break

        if not discard:
            filtered.append((i, obj.tolist(), model))

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering 0.0 values and thresholds.")

    indices, filtered_objs, filtered_models = zip(*filtered)
    medians = [np.median(obj) for obj in filtered_objs]
    best_idx = int(np.argmin(medians))

    return {
        "index": indices[best_idx],
        "score": float(medians[best_idx]),
        "objectives": filtered_objs[best_idx],
        "models": filtered_models[best_idx]
    }


def select_weighted_solution(
    objectives: npt.NDArray[np.float64],
    models: List[Any],
    weights: npt.NDArray[np.float64] | List[float]
) -> Dict[str, Any]:
    """
    Returns the solution with the lowest weighted average of objectives.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must match.")

    num_obj = objectives.shape[1]
    weights_arr = np.array(weights, dtype=float)
    if len(weights_arr) != num_obj:
        raise ValueError("Weights length must match number of objectives.")

    # Filter zeros
    filtered = [
        (i, obj.tolist(), model)
        for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.isclose(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering 0.0 objective values.")

    indices, filtered_objs, filtered_models = zip(*filtered)

    # Normalize weights
    weights_arr /= weights_arr.sum()
    
    # Calculate weighted score
    weighted_means = [float(np.dot(obj, weights_arr)) for obj in filtered_objs]
    best_idx = int(np.argmin(weighted_means))
    original_idx = indices[best_idx]

    return {
        "index": original_idx,
        "score": weighted_means[best_idx],
        "objectives": filtered_objs[best_idx],
        "model": filtered_models[best_idx]
    }