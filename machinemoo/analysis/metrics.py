import numpy as np
from pymoo.indicators.hv import HV
from typing import List, Optional

def compute_hypervolume_progress(
    solutions: List[np.ndarray],
    reference_point: Optional[np.ndarray] = None
) -> List[float]:
    """
    Calculates the cumulative hypervolume values over the list of solutions.

    Args:
        solutions: A list of objective vectors (solutions).
        reference_point: Optional reference point for the hypervolume. If not provided,
                         uses a vector of ones with appropriate dimensions.

    Returns:
        A list of hypervolume values computed incrementally.
    """
    if reference_point is None:
        reference_point = np.ones_like(solutions[0])
    
    hv_indicator = HV(ref_point=reference_point)

    hypervolume_values = []
    current_solution_set = []

    for solution in solutions:
        current_solution_set.append(solution)
        hv_value = hv_indicator(np.array(current_solution_set))
        hypervolume_values.append(hv_value)

    return hypervolume_values