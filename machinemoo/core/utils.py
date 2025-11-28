import numpy as np
import numpy.typing as npt

def normf(
    obj: npt.NDArray[np.float64],
    global_lower: npt.NDArray[np.float64],
    global_upper: npt.NDArray[np.float64],
    norm: bool = True
) -> npt.NDArray[np.float64]:
    """
    Normalize the objective vector based on global lower and upper bounds.

    Args:
        obj (npt.NDArray[np.float64]): Objective vector to be normalized.
        global_lower (npt.NDArray[np.float64]): Global lower bounds (utopia point).
        global_upper (npt.NDArray[np.float64]): Global upper bounds (nadir point).
        norm (bool): Whether to perform normalization (scale to [0,1]). 
                     If False, only shifts by lower bound (translation).

    Returns:
        npt.NDArray[np.float64]: Normalized objective vector.
    """
    if norm:
        # Avoid division by zero if upper == lower (though usually avoided by logic elsewhere)
        denominator = global_upper - global_lower
        return (obj - global_lower) / denominator
    else:
        return obj - global_lower


def normw(
    w: npt.NDArray[np.float64],
    global_lower: npt.NDArray[np.float64],
    global_upper: npt.NDArray[np.float64],
    norm: bool = True
) -> npt.NDArray[np.float64]:
    """
    Normalize the weight vector based on the scale of objectives.
    
    Adjusts weights so that they account for the different scales of the 
    objectives (defined by global_upper - global_lower).

    Args:
        w (npt.NDArray[np.float64]): Original weighting vector.
        global_lower (npt.NDArray[np.float64]): Global lower bounds.
        global_upper (npt.NDArray[np.float64]): Global upper bounds.
        norm (bool): Whether to normalize the weights.

    Returns:
        npt.NDArray[np.float64]: Normalized weighting vector.
    """
    if norm:
        w_ = w * (global_upper - global_lower)
        # Handle case where sum is 0 to avoid NaN, though unlikely with proper weights
        w_sum = w_.sum()
        if w_sum == 0:
            return w_
        return w_ / w_sum
    else:
        return w