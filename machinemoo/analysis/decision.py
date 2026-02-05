import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, Tuple, Dict
import pandas as pd
from pymcdm.methods import TOPSIS


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