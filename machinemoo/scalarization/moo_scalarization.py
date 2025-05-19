from typing import Callable, Union, Tuple, Self, Any, Optional
#from typing_extensions import Self
import numpy as np
import numpy.typing as npt

from machinemoo.scalarization.scalarization_interface import scalar_interface, single_interface, w_interface

class Scalarization(w_interface, single_interface, scalar_interface):
    """
    Base class for scalarization
    """
    def __init__(self, num_objs, *args, **kwargs) -> None:
        """
        Initialize internal the Scalarization state.
        """
        self.__M: int = num_objs
        self.__x: Any = None
        self.__gradient = None
        self.__w: Optional[npt.NDArray[np.float64]] = None
        self.__objs: Optional[npt.NDArray[np.float64]] = None
        #self.model: Any = None

    @property
    def M(self) -> int:
        return self.__M

    @property
    def feasible(self) -> bool:
        return True

    @property
    def optimum(self) -> bool:
        return True

    @property
    def objs(self) -> npt.NDArray[np.float64]:
        #assert self.__objs is not None
        return self.__objs

    @property
    def x(self) -> Any:
        return self.__x

    @property
    def w(self) -> npt.NDArray[np.float64]:
        #assert self.__w is not None
        return self.__w
    
    @property
    def gradient(self) -> Optional[Any]:
        return self.__gradient
    
    def training(self, w: Union[int, npt.NDArray[np.float64]]) -> tuple[Any, npt.NDArray[np.float64], Optional[Any]]:
        raise NotImplementedError
    # training: Callable[..., Any] = training #may needed to fix mypy error

    def optimize(self, w: Union[int, npt.NDArray[np.float64]]) -> Self:
        """
        Performs multi-objective scalarization using a weight vector.

        Parameters:
            w: Either an integer (index) or a 1D numpy array of shape (M,)

        Returns:
            self
        """
        if isinstance(w, int):
            self.__w = np.zeros(self.M)
            self.__w[w] = 1
        elif isinstance(w, np.ndarray) and w.ndim == 1 and w.size == self.M:
            self.__w = w
        else:
            raise ValueError("w is in the wrong format")

        result  = self.training(self.__w)

        if len(result) == 2:
            model, self.__objs = result
            self.__gradient = None
        elif len(result) == 3:
            model, self.__objs, self.__gradient = result
        else:
            raise ValueError("training() must return a tuple of 2 or 3 elements")

        self.__M = len(self.__objs)
        self.__x = model
        #self.__gradient = gradient[0] if gradient else None
        return self

class MooScalarization(w_interface, single_interface, scalar_interface):
    def __init__(
        self,
        model: object,
        train_fn: Callable[[object, npt.NDArray[np.float64]], Tuple[npt.NDArray[np.float64], object]],
        num_objs: int = 2
    ) -> None:
        """
        Initializes the MooScalarization object.

        Parameters:
            model: An initialized model
            train_fn: A function that trains the model given a weight vector and 
                      returns a tuple (objectives, trained_model)
            num_objs: Number of objectives (M)
        """
        self.model: object = model
        self.train: Callable[[object, npt.NDArray[np.float64]], Tuple[npt.NDArray[np.float64], object]] = train_fn
        self.__M: int = num_objs
        self.__objs: npt.NDArray[np.float64] = np.zeros(num_objs)
        self.__x: object = None 
        self.__w: Union[int, npt.NDArray[np.float64]] = None
        self.__gradient = None

    @property
    def M(self) -> int:
        return self.__M

    @property
    def feasible(self) -> bool:
        return True

    @property
    def optimum(self) -> bool:
        return True

    @property
    def objs(self) -> npt.NDArray[np.float64]:
        return self.__objs

    @property
    def x(self) -> object:
        return self.__x

    @property
    def w(self) -> npt.NDArray[np.float64]:
        return self.__w
    
    @property
    def gradient(self):
        return self.__gradient

    def optimize(self, weight: Union[int, npt.NDArray[np.float64]]) -> "MooScalarization":
        """
        Performs multi-objective scalarization using a weight vector.

        Parameters:
            w: Either an integer (index) or a 1D numpy array of shape (M,)

        Returns:
            self
        """
        if isinstance(weight, int):
            self.__w = np.zeros(self.M)
            self.__w[weight] = 1
        elif isinstance(weight, np.ndarray) and weight.ndim == 1 and weight.size == self.M:
            self.__w = weight
        else:
            raise ValueError("w is in the wrong format")

        self.__objs, self.model, gradients = self.train(self.model, self.__w)
        self.__x = self.model
        self.__gradient = gradients
        return self
