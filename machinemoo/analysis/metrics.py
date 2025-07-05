import numpy as np
import numpy.typing as npt
from pymoo.indicators.hv import HV

def compute_hypervolume_progress(
    solutions: npt.NDArray[np.float64],
    reference_point: npt.NDArray[np.float64] = np.array([])
) -> list[float]:
    """
    Calculates the cumulative hypervolume values over the list of solutions.

    Args:
        solutions: A list of objective vectors (solutions).
        reference_point: Optional reference point for the hypervolume. If not provided,
                         uses a vector of ones with appropriate dimensions.

    Returns:
        A list of hypervolume values computed incrementally.
    """
    if len(reference_point) == 0:
        reference_point = np.ones_like(solutions[0])
    
    hv_indicator = HV(ref_point=reference_point)

    hypervolume_values = []
    current_solution_set = []

    for solution in solutions:
        current_solution_set.append(solution)
        hv_value = hv_indicator(np.array(current_solution_set))
        hypervolume_values.append(hv_value)

    return hypervolume_values