import numpy as np
import numpy.typing as npt
from typing import Any
from collections.abc import Callable
from typing_extensions import Self

from machinemoo import scalar_interface, single_interface, w_interface

class Scalarization(w_interface, single_interface, scalar_interface):
    """Base class for scalarization strategies in multi-objective optimization.

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
        """Initializes the scalarization state.

        Args:
            num_objs (int): Number of objectives in the optimization problem.
        """
        self.__M: int = num_objs
        self.__x: Any = None
        self.__gradient = None
        self.__w: np.float64 = np.zeros(num_objs)
        self.__objs: npt.NDArray[np.float64] = np.zeros( num_objs)
        self.lower_bound_estimate = lower_bound_estimate
        self.__objs_lower = None
        #self.model: Any = None

    @property
    def M(self) -> int:
        """Returns the number of objectives.

        Returns:
            int: Number of objectives.
        """
        return self.__M

    @property
    def feasible(self) -> bool:
        """Indicates whether the current solution is feasible.

        Returns:
            bool: Always True for this base implementation.
        """
        return True

    @property
    def optimum(self) -> bool:
        """Indicates whether the optimal solution has been reached.

        Returns:
            bool: Always True for this base implementation.
        """
        return True

    @property
    def objs(self) -> npt.NDArray[np.float64]:
        """Returns the objective vector of the solutions.

        Returns:
            np.ndarray: Objective values for each objective function.
        """
        return self.__objs

    def _compute_lipschitz(self):
        raise NotImplementedError("Subclass must implement Lipschitz calculation")
    
    @property
    def objs_lower(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values.
        
        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        if self.__objs_lower is not None:
            return self.__objs_lower

        if self.lower_bound_estimate == "lipschitz":
            self.__objs_lower = self._compute_lipschitz()
        elif self.lower_bound_estimate == "zero":
            self.__objs_lower = np.zeros(self.__M)
        else:
            self.__objs_lower = self.objs - self.lower_bound_estimate * abs(self.objs)
        return self.__objs_lower

    @property
    def x(self) -> Any:
        """Returns the learned model or decision variables.

        Returns:
            Any: Learned model or solution representation.
        """
        return self.__x

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """Returns the weight vector used in scalarization.

        Returns:
            np.ndarray: Scalarization weights.
        """
        return self.__w
    
    @property
    def gradient(self) -> Any | None:
        """Returns the gradient of the scalarized objective, if available.

        Returns:
            Any or None: Gradient vector or None if not computed.
        """
        return self.__gradient
    
    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[Any, npt.NDArray[np.float64], Any] | tuple[Any, npt.NDArray[np.float64]]:
        """Trains a model or solves the scalarized problem given a weight vector.

        This method must be implemented by subclasses.

        Args:
            weight (np.ndarray): Weight vector for scalarization.

        Returns:
            tuple: A tuple containing the trained model, objective values, and optionally the gradient.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError
    #training: Callable[..., Any] = training #may needed to fix mypy error

    def optimize(self, weight: int | npt.NDArray[np.float64]) -> Self:
        """Performs scalarization optimization with a given weight vector.

        The method trains a model or solves a problem with the specified weights,
        storing the result as internal state.

        Args:
            weight (int | npt.NDArray[np.float64]): Weight index (for single-objective) or full weight vector.

        Returns:
            Self: The scalarization instance with updated solution.
        """
        if isinstance(weight, int):
            self.__w = np.zeros(self.M)
            self.__w[weight] = 1
        elif isinstance(weight, np.ndarray) and weight.ndim == 1 and weight.size == self.M:
            self.__w = weight
        else:
            raise ValueError("w is in the wrong format")

        self.__objs_lower = None
        result: tuple[Any, Any, Any | None] = self.training(self.__w)

        if len(result) == 2:
            model, self.__objs = result
            self.__gradient = None
        elif len(result) == 3:
            model, self.__objs, self.__gradient = result
        else:
            raise ValueError("training() must return a tuple of 2 or 3 elements")

        self.__M = len(self.__objs)
        self.__x = model

        return self

class LipschitzTorchMixin:
    """Mixin that provides a Lipschitz calculation using PyTorch."""

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values with lipschitz estamation.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        w_gradient = self.w@np.array([self.gradient[idx] for idx in range(self.M)])
        objs_delta = 1/(2*self.w@self.L)*w_gradient@w_gradient
        self.__objs_lower = self.objs - objs_delta
        return self.__objs_lower

class LipschitzRegLoghMixin:
    """Mixin that provides a Lipschitz calculation using Logistic Regression from Scikit Learn."""

    def _compute_lipschitz(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values with lipschitz estamation.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        J = np.array(self.gradient)
        grad_w = self.w@J
        L = self.w@self.L
        objs_delta = 1/L*J@grad_w - 1/2*self.L/(L**2)*(grad_w@grad_w)
        self.__objs_lower = self.objs - objs_delta
        return self.__objs_lower

class MooScalarization(w_interface, single_interface, scalar_interface):
    """Implements scalarization for machine learning models using an external training function.

    This class is designed to support model training guided by scalarization,
    where a `train_fn` receives a model and a weight vector and returns objectives,
    the updated model, and optionally gradients.
    """
    def __init__(
        self,
        model: object,
        train_fn: Callable[[Any, npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], Any, Any | None]],
        num_objs: int = 2
    ) -> None:
        """Initializes the MooScalarization instance.

        Args:
            model (object): Initial machine learning model or structure to be optimized.
            train_fn (Callable): A function that trains the model with a weight vector
                and returns (objectives, updated model, [optional gradient]).
            num_objs (int, optional): Number of objective functions. Defaults to 2.
        """
        self.model: object = model
        self.train: Callable[[Any, npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], Any, Any | None]] = train_fn
        self.__M: int = num_objs
        self.__objs: npt.NDArray[np.float64] = np.zeros(num_objs)
        self.__x: object = None 
        self.__w: npt.NDArray[np.float64] = np.zeros(num_objs)
        self.__gradient = None

    @property
    def M(self) -> int:
        """Returns the number of objectives.

        Returns:
            int: Number of objectives.
        """
        return self.__M

    @property
    def feasible(self) -> bool:
        """Indicates whether the current solution is feasible.

        Returns:
            bool: Always True for this base implementation.
        """
        return True

    @property
    def optimum(self) -> bool:
        """Indicates whether the optimal solution has been reached.

        Returns:
            bool: Always True for this base implementation.
        """
        return True

    @property
    def objs(self) -> npt.NDArray[np.float64]:
        """Returns the objective vector of the solutions.

        Returns:
            np.ndarray: Objective values for each objective function.
        """
        return self.__objs
    
    @property
    def objs_lower(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values.
        
        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        return self.__objs_lower

    @property
    def x(self) -> Any:
        """Returns the learned model or decision variables.

        Returns:
            Any: Learned model or solution representation.
        """
        return self.__x

    @property
    def w(self) -> npt.NDArray[np.float64]:
        """Returns the weight vector used in scalarization.

        Returns:
            np.ndarray: Scalarization weights.
        """
        return self.__w
    
    @property
    def gradient(self) -> Any | None:
        """Returns the gradient of the scalarized objective, if available.

        Returns:
            Any or None: Gradient vector or None if not computed.
        """
        return self.__gradient

    def optimize(self, weight: int | npt.NDArray[np.float64]) -> Self:
        """Performs optimization using the provided weight vector.

        Trains the underlying model with the weight vector, stores the
        objectives, model, and optionally gradient.

        Args:
            weight (int | npt.NDArray[np.float64]): Index of the objective (int) or a weight vector (np.ndarray).

        Returns:
            Self: The scalarization instance with updated model and objectives.
        """
        if isinstance(weight, int):
            self.__w = np.zeros(self.M)
            self.__w[weight] = 1
        elif isinstance(weight, np.ndarray) and weight.ndim == 1 and weight.size == self.M:
            self.__w = weight
        else:
            raise ValueError("w is in the wrong format")

        self.__objs, self.model, self.__gradient = self.train(self.model, self.__w)
        self.__x = self.model

        return self