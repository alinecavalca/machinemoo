import numpy as np
import numpy.typing as npt
from typing import Any, Tuple, Optional, Union
from collections.abc import Callable
from typing_extensions import Self

from machinemoo.scalarization.scalarization_interface import scalar_interface, w_interface, single_interface
from machinemoo.moo.fix_weight import solve_optimal_w_active_set#, solve_optimal_w_scipy

class BaseScalarizer(w_interface, single_interface, scalar_interface):
    """
    Base class for scalarization strategies in multi-objective optimization.

    This class defines a generic interface for performing scalarization using a 
    weight vector. It serves as a base for scalarization methods that learn models 
    or compute solutions using scalarized objectives.
    """
    def __init__(
            self,
            num_objs: int,
            lower_bound_estimate: str | float = "zero", 
            *args: Any, 
            **kwargs: Any
        ) -> None:
        """
        Initializes the scalarization state.

        Args:
            num_objs (int): Number of objectives in the optimization problem.
            lower_bound_estimate (str | float): Strategy for lower bound estimation.
                                                Options: "zero", "lipschitz", or a float factor.
        """
        self._M: int = num_objs
        self._x: Any = None
        self._gradient: Optional[npt.NDArray[np.float64]] = None
        self._w: npt.NDArray[np.float64] = np.zeros(num_objs)
        self._objs: npt.NDArray[np.float64] = np.zeros(num_objs)
        
        self.lower_bound_estimate = lower_bound_estimate
        self._objs_lb: Optional[npt.NDArray[np.float64]] = None
        
        # Lipschitz constants vector (to be set by subclasses or mixins)
        self.L: Optional[npt.NDArray[np.float64]] = None 

    @property
    def M(self) -> int:
        """Returns the number of objectives."""
        return self._M

    @property
    def feasible(self) -> bool:
        """Indicates whether the current solution is feasible."""
        return True

    @property
    def optimum(self) -> bool:
        """Indicates whether the optimal solution has been reached."""
        return True

    @property
    def objs(self) -> npt.NDArray[np.float64]:
        """Returns the objective vector of the solutions."""
        return self._objs
    
    def objective(self) -> npt.NDArray[np.float64]:
        """
        Returns the computed objective values.
        Implementation of the abstract method from single_interface.
        """
        return self.objs

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """
        Computes the Lipschitz-based lower bound estimate.
        Must be implemented by subclasses or provided via Mixins.
        """
        raise NotImplementedError("Subclass must implement Lipschitz calculation or use a Mixin.")
    
    @property
    def objs_lb(self) -> npt.NDArray[np.float64]:
        """
        Returns a lower estimate of the objective values.
        
        Uses the strategy defined by `lower_bound_estimate`.
        
        Returns:
            np.ndarray: Lower estimate for each objective function.
        """
        if self._objs_lb is not None:
            return self._objs_lb

        if self.lower_bound_estimate == "lipschitz":
            self._objs_lb = self._compute_lipschitz()
        elif self.lower_bound_estimate == "zero":
            self._objs_lb = np.zeros(self._M)
        else:
            # Assume it's a float factor
            factor = float(self.lower_bound_estimate)
            self._objs_lb = np.asarray(self.objs, dtype=np.float64) - np.multiply(factor, np.abs(self.objs))
            
        return self._objs_lb

    @property
    def x(self) -> Any:
        """Returns the learned model or decision variables."""
        return self._x

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """Returns the weight vector used in scalarization."""
        return self._w
    
    @property
    def gradient(self) -> Optional[npt.NDArray[np.float64]]:
        """Returns the gradient of the scalarized objective, if available."""
        return self._gradient
    
    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> Union[Tuple[Any, npt.NDArray[np.float64]], Tuple[Any, npt.NDArray[np.float64], Any]]:
        """
        Trains a model or solves the scalarized problem given a weight vector.

        This method must be implemented by subclasses.

        Args:
            weight (np.ndarray): Weight vector for scalarization.

        Returns:
            tuple: (trained_model, objective_values) OR (trained_model, objective_values, gradient)
        """
        raise NotImplementedError("Subclasses must implement the training method.")

    def optimize(self, weight: Union[int, npt.NDArray[np.float64]]) -> Self:
        """
        Performs scalarization optimization with a given weight vector.

        Args:
            weight (int | np.ndarray): Weight index (for single-objective) or full weight vector.

        Returns:
            Self: The scalarization instance with updated solution.
        """
        # Handle weight input
        if isinstance(weight, (int, np.integer)):
            self._w = np.zeros(self.M)
            self._w[int(weight)] = 1.0
        elif isinstance(weight, np.ndarray) and weight.ndim == 1 and weight.size == self.M:
            self._w = weight.astype(np.float64)
        elif isinstance(weight, list) and len(weight) == self.M:
            self._w = np.array(weight, dtype=np.float64)
        else:
            raise ValueError(f"Weight must be an int index or an array of size M ({self.M}). Got: {weight}")

        # Reset bounds cache
        self._objs_lb = None
        
        # Execute training
        result = self.training(self._w)

        if len(result) == 2:
            model, objs = result
            self._gradient = None
        elif len(result) == 3:
            model, objs, gradient = result
            self._gradient = gradient
            self._w = solve_optimal_w_active_set(np.array(gradient), self.L)
        else:
            raise ValueError("training() must return a tuple of 2 or 3 elements (model, objs, [gradient])")

        self._objs = np.array(objs)
        self._x = model
        
        # Update M in case training changed dimensions (though unusual)
        if len(self._objs) != self._M:
            self._M = len(self._objs)

        return self


class LipschitzTorchMixin:
    """Mixin that provides a Lipschitz calculation for PyTorch-based scalarizations."""

    # Mixin expects these attributes from the host class
    gradient: Optional[npt.NDArray[np.float64]]
    L: Optional[npt.NDArray[np.float64]]
    w: npt.NDArray[np.float64]
    objs: npt.NDArray[np.float64]

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """
        Returns a lower estimate of the objective values using Lipschitz constants.
        """
        if self.gradient is None or self.L is None:
            return self.objs

        grads = np.array(self.gradient)
        w_gradient = self.w @ grads 
        w_L = self.w @ self.L
        
        if w_L <= 1e-9:
            return self.objs

        # Quadratic correction
        # Note: This assumes specific structure of gradients for the correction term
        objs_delta = (1/w_L) * grads @ w_gradient - (1/2) * (self.L / (w_L**2)) * (w_gradient @ w_gradient)
        return np.asarray(self.objs, dtype=np.float64) - objs_delta


class LipschitzRegLoghMixin:
    """Mixin that provides a Lipschitz calculation using Logistic Regression logic."""

    # Mixin expects these attributes from the host class
    gradient: Optional[npt.NDArray[np.float64]]
    L: Optional[npt.NDArray[np.float64]]
    w: npt.NDArray[np.float64]
    objs: npt.NDArray[np.float64]

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """
        Returns a lower estimate of the objective values with Lipschitz estimation
        specific to Logistic Regression (linear models).
        """
        if self.gradient is None or self.L is None:
            return self.objs

        J = np.array(self.gradient) # Jacobian-like
        grad_w = self.w @ J
        L_scalar = self.w @ self.L
        
        if L_scalar == 0:
            return self.objs

        # Formula specific to linear models/logistic regression
        objs_delta = (1/L_scalar) * (J @ grad_w) - (1/2) * (self.L / (L_scalar**2)) * (grad_w @ grad_w)
        
        return self.objs - objs_delta


class FunctionalScalarizer(BaseScalarizer):
    """
    Implements scalarization for machine learning models using an external training function.
    
    This replaces the old 'MooScalarization' name to clearer indicate its purpose:
    it wraps a functional training routine.
    """
    def __init__(
        self,
        model: Any,
        train_fn: Callable[
            [Any, npt.NDArray[np.float64]],
            Union[
                Tuple[npt.NDArray[np.float64], Any],
                Tuple[npt.NDArray[np.float64], Any, Optional[Any]]
            ]
        ],
        num_objs: int = 2,
        lipschitz_constants: Optional[npt.NDArray[np.float64]] = None,
        **kwargs: Any
    ) -> None:
        """
        Args:
            model (Any): Initial machine learning model.
            train_fn (Callable): Function: (model, weights) -> (objectives, updated_model, [gradient]).
            num_objs (int): Number of objectives.
            lipschitz_constants (np.ndarray, optional): Array of Lipschitz constants.
        """
        estimate_type = kwargs.pop("lower_bound_estimate", "lipschitz" if lipschitz_constants is not None else "zero")
        
        super().__init__(num_objs=num_objs, lower_bound_estimate=estimate_type, **kwargs)

        self.model = model
        self.train_fn = train_fn
        self.L = np.array(lipschitz_constants, dtype=np.float64) if lipschitz_constants is not None else None

    def training(self, weight: npt.NDArray[np.float64]) -> Union[Tuple[Any, npt.NDArray[np.float64]], Tuple[Any, npt.NDArray[np.float64], Optional[Any]]]:
        """Delegates training to the provided function."""
        result = self.train_fn(self.model, weight)

        if not isinstance(result, (tuple, list)):
            raise ValueError("train_fn must return a tuple: (objs, model) or (objs, model, gradient)")

        if len(result) == 3:
            objs, model, grad = result
            objs_arr = np.asarray(objs, dtype=np.float64)
            self.model = model
            return model, objs_arr, grad
        elif len(result) == 2:
            objs, model = result
            objs_arr = np.asarray(objs, dtype=np.float64)
            self.model = model
            return model, objs_arr
        else:
            raise ValueError("train_fn must return (objs, model) or (objs, model, gradient)")

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """Generic Lipschitz estimation using quadratic approximation."""
        if self.gradient is None or self.L is None:
            return self.objs

        grads = np.array(self.gradient)
        w_gradient = self.w @ grads
        w_L = self.w @ self.L

        if w_L <= 1e-9:
            return self.objs

        objs_delta = (1.0 / (2.0 * w_L)) * np.dot(w_gradient, w_gradient)
        return np.asarray(self.objs, dtype=np.float64) - objs_delta