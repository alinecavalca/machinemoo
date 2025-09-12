import numpy as np
import numpy.typing as npt
from typing import Any
import pandas as pd
from pymcdm.methods import TOPSIS

def topsis_model_selection(losses, metrics=None, criteria_types=None, weights=None, model_names=None):
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
    n_criteria = data.shape[1]
    if criteria_types is None:
        # losses = -1, métricas = 1
        criteria_types = [-1]*losses.shape[1]
        if metrics:
            criteria_types += [1]*len(metrics)

    # Pesos
    if weights is None:
        weights = np.ones(n_criteria) / n_criteria

    # Nomes de modelos
    if model_names is None:
        model_names = [f"M{i}" for i in range(n_models)]

    # Aplica TOPSIS
    topsis = TOPSIS()
    scores = topsis(data, weights, criteria_types)

    # Ranking (ordenando)
    ranking_order = np.argsort(-scores)  # do maior score para o menor
    df = pd.DataFrame({
        "Model": [model_names[i] for i in ranking_order],
        "TOPSIS Score": scores[ranking_order],
        "Original Index": ranking_order
    })
    df["Rank"] = np.arange(1, n_models + 1)

    # Melhor modelo (primeiro do ranking)
    best_index = int(ranking_order[0])  # índice original na lista de entrada
    best_model_name = model_names[best_index]
    best_losses = losses[best_index]

    return df, best_index, best_model_name, best_losses

def select_lowest_median_solution(
    objectives: npt.NDArray[np.float64],
    models: list[Any],
    objective_thresholds: dict[int, float] | None = None
) -> dict[str, int | float | list[float] | Any]:
    """
    Returns the solution with the lowest median objective value, optionally filtering by thresholds.

    This function filters out any solutions containing zero-valued objectives
    and selects the one with the lowest median among the remaining.
    It can also discard solutions exceeding a threshold for specific objectives.

    Args:
        objectives (np.ndarray): An array where each row is a vector of objective values for a solution.
        models (list): A list of trained models or objects corresponding to each solution.
        objective_thresholds (dict[int, float], optional): Dictionary of {objective_index: max_allowed_value}.
    
    Returns:
        dict:
            'index' (int): Index of the selected solution in the original array.
            'score' (float): Median of the objectives of the selected solution.
            'objectives' (List[float]): Objective values of the selected solution.
            'models' (Any): Model corresponding to the selected solution.
    
    Raises:
        ValueError: If the number of models does not match the number of objective vectors.
        ValueError: If no valid solution remains after filtering.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must match.")

    filtered = []
    for i, (obj, model) in enumerate(zip(objectives, models)):
        if np.any(np.isclose(obj, 0.0)):
            continue

        discard = False
        if objective_thresholds:
            for idx, thresh in objective_thresholds.items():
                val = float(obj[idx])
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
    models: list[Any],
    weights: npt.NDArray[np.float64] | list[float] | list[int]
) -> dict[str, int | float | list[float] | Any]:
    """
    Returns the solution with the lowest weighted average of objectives, using a given weight vector.

    This function filters out solutions containing near-zero objective values,
    normalizes the weights, and computes the weighted mean of each solution.

    Args:
        objectives (np.ndarray): An array where each row contains objective values for a solution.
        models (list): A list of trained models or associated objects, one per solution.
        weights (np.ndarray): A 1D array of weights, with length equal to the number of objectives.

    Returns:
        dict:
            'index' (int): Index of the selected solution in the original array.
            'score' (float): Weighted mean of the objectives of the selected solution.
            'objectives' (List[float]): Objective values of the selected solution.
            'model' (Any): Models corresponding to the selected solution.
    Raises:
        ValueError: If the number of models and objectives differ.
        ValueError: If the weights vector length does not match the number of objectives.
        ValueError: If no valid solution remains after filtering.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must match.")

    num_obj = objectives.shape[1]
    weights_arr = np.array(weights, dtype=float)
    if len(weights_arr) != num_obj:
        raise ValueError("Weights length must match number of objectives.")

    filtered = [
        (i, obj.tolist(), model)
        for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.isclose(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering 0.0 objective values.")

    indices, filtered_objs, filtered_models = zip(*filtered)

    weights_arr /= weights_arr.sum()
    weighted_means = [float(np.dot(obj, weights_arr)) for obj in filtered_objs]
    best_idx = int(np.argmin(weighted_means))
    original_idx = indices[best_idx]

    return {
        "index": original_idx,
        "score": weighted_means[best_idx],
        "objectives": filtered_objs[best_idx],
        "model": filtered_models[best_idx]
    }

## TOPSIS