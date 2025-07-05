import numpy as np
import numpy.typing as npt

from typing import Any

def get_best_median_solution(
    objectives: npt.NDArray[np.float64],
    models: list[Any]
) -> tuple[list[float], Any, float, int]:
    """
    Returns the solution with the lowest median objective value, along with the corresponding model.

    This function filters out any solutions containing zero-valued objectives
    and selects the one with the lowest median among the remaining.

    Args:
        objectives (np.ndarray): An array where each row is a vector of objective values for a solution.
        models (list): A list of trained models or objects corresponding to each solution.

    Returns:
        tuple:
            - list[float]: Objective vector with the lowest median (after filtering).
            - Any: Corresponding model.
            - float: Median value of the selected solution.
            - int: Index of the selected solution in the original list.

    Raises:
        ValueError: If the number of models does not match the number of objective vectors.
        ValueError: If no valid solution remains after filtering.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must be the same.")

    # Filter out solutions that contain 0.0
    filtered = [
        (i, obj, model) for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.equal(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering out 0.0 objective values.")
    
    # Unpack filtered data
    indices, filtered_objs, filtered_models = zip(*filtered)

    medians = [np.median(obj) for obj in filtered_objs]
    best_idx_in_filtered = int(np.argmin(medians))
    original_idx = indices[best_idx_in_filtered]

    return (
        filtered_objs[best_idx_in_filtered],
        filtered_models[best_idx_in_filtered],
        medians[best_idx_in_filtered],
        original_idx
    )

def get_best_weighted_solution(
    objectives: npt.NDArray[np.float64],
    models: list[Any],
    weights: npt.NDArray[np.float64] | list[float]
) -> tuple[list[float], Any, float, int]:
    """
    Returns the solution with the lowest weighted average of objectives, using a given weight vector.

    This function filters out solutions containing near-zero objective values,
    normalizes the weights, and computes the weighted mean of each solution.

    Args:
        objectives (np.ndarray): An array where each row contains objective values for a solution.
        models (list): A list of trained models or associated objects, one per solution.
        weights (np.ndarray): A 1D array of weights, with length equal to the number of objectives.

    Returns:
        tuple:
            - list[float]: Objective vector with the lowest weighted mean.
            - Any: Corresponding model.
            - float: Weighted mean value of the selected solution.
            - int: Index of the selected solution in the original list.

    Raises:
        ValueError: If the number of models and objectives differ.
        ValueError: If the weights vector length does not match the number of objectives.
        ValueError: If no valid solution remains after filtering.
    """
    if len(objectives) != len(models):
        raise ValueError("Length of objectives and models must be the same.")
    
    num_objectives = len(objectives[0])
    if any(len(obj) != num_objectives for obj in objectives):
        raise ValueError("All objective vectors must have the same length.")
    
    if len(weights) != num_objectives:
        raise ValueError("Weights vector must have the same length as each objective vector.")

    # Filter out solutions that contain 0.0 (or are close to it)
    filtered = [
        (i, obj, model) for i, (obj, model) in enumerate(zip(objectives, models))
        if not np.any(np.isclose(obj, 0.0))
    ]

    if not filtered:
        raise ValueError("No valid solutions remaining after filtering out 0.0 objective values.")

    # Unpack filtered data
    indices, filtered_objs, filtered_models = zip(*filtered)

    weights = np.array(weights)
    # Normalize to sum to 1
    weights = weights / weights.sum()

    weighted_means = [np.dot(obj, weights) for obj in filtered_objs]
    best_idx_in_filtered = int(np.argmin(weighted_means))
    original_idx = indices[best_idx_in_filtered]

    return (
        filtered_objs[best_idx_in_filtered],
        filtered_models[best_idx_in_filtered],
        weighted_means[best_idx_in_filtered],
        original_idx
    )