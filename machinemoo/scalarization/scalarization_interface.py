# -*- coding: utf-8 -*-
"""
Scalarization Interface

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause

import numpy as np
import numpy.typing as npt
from typing import Any, Self
from abc import ABCMeta, abstractmethod

__all__ = [
    "scalar_interface",
    "w_interface",
    "single_interface"
]

class scalar_interface(metaclass=ABCMeta):
    """Abstract base interface for scalarization methods in multi-objective optimization."""
    # - propertys
    @property
    @abstractmethod
    def M(self) -> int:
        """
        Abstract property to get the number of objectives (M) in the problem.
        
        Returns:
            int: The number of objectives in the problem.
        """
        pass

    @property
    @abstractmethod
    def feasible(self) -> bool:
        """
        Abstract property to idicates whether the current solution is feasible.
        
        Returns:
            bool: True if the current solution is feasible.
        """
        pass

    @property
    @abstractmethod
    def optimum(self) -> bool:
        """
        Abstract property to indicates whether the optimal solution has been reached.
        
        Returns:
            bool: True if the optimum has been reached.
        """
        pass

    @property
    @abstractmethod
    def objs(self) -> npt.NDArray[np.float64]:
        """
        Abstract property to get the objectives of the problem.
        
        Returns:
            np.ndarray: Objective values for each objective function.
        """
        pass

    @property
    @abstractmethod
    def x(self) -> Any:
        """
        Abstract property to get the decision variables of the problem.
        
        Returns:
            Any: The decision variables of the problem.
        """
        pass

    @abstractmethod
    def optimize(self, *args: Any) -> Self:
        """
        Abstract method to executes the scalarization optimization procedure.

        Parameters:
            *args: Arguments needed to be used in the optimization process.

        Returns:
            Self: The instance after optimization.
        """
        pass


class w_interface(scalar_interface, metaclass=ABCMeta):
    """Abstract interface for scalarizations using weight vectors."""
    # - propertys
    @property
    @abstractmethod
    def w(self) -> int | npt.NDArray[np.float64]:
        """
        Abstract property to get the weights of the problem.
        
        Returns:
            int or np.ndarray: Scalarization weights (index or vector).
        """
        pass


class single_interface(scalar_interface, metaclass=ABCMeta):
    """Abstract interface for single-objective optimization methods."""
    # - propertys
    @property
    @abstractmethod
    def w(self) -> int | npt.NDArray[np.float64]: 
        """
        Abstract property to get the weights of the problem.
        
        Returns:
            int or np.ndarray: Scalarization weights (index or vector).
        """
        pass

    def objetive(self) -> npt.NDArray[np.float64]:
        """
        Abstract method to define the objective to be optimized.
        
        Returns:
            np.ndarray: The computed objective values.
        """
        pass


class box_interface(metaclass=ABCMeta):
    """Abstract interface for box-constrained problems."""
    ## - propertys
    @property
    @abstractmethod
    def u(self) -> Any:
        """
        Abstract property to get the upper bound of the problem.
        
        Returns:
            Any: Upper bound.
        """
        pass

    @property
    @abstractmethod
    def l(self) -> Any:
        """
        Abstract property to get the lower bound of the problem.
        
        Returns:
            Any: Lower bound.
        """
        pass

    @property
    @abstractmethod
    def c(self) -> Any:
        """
        Abstract property to get the constraints of the problem.
        
        Returns:
            Any: The constraints of the problem.
        """
        pass
