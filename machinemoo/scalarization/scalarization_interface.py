# -*- coding: utf-8 -*-
"""
Scalarization Interface

Author: Marcos M. Raimundo <marcosmrai@gmail.com>
        Laboratory of Bioinformatics and Bioinspired Computing
        FEEC - University of Campinas
"""
# License: BSD 3 clause

from abc import ABCMeta, abstractmethod

__all__ = [
    "scalar_interface",
    "w_interface",
    "single_interface"
]

class scalar_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def M(self):
        """
        Abstract property to get the number of objectives (M) in the problem.
        
        Returns:
            int: The number of objectives in the problem.
        """
        pass

    @property
    @abstractmethod
    def feasible(self):
        """
        Abstract property to get the feasible region of the problem.
        
        Returns:
            A representation of the feasible region.
        """
        pass

    @property
    @abstractmethod
    def optimum(self):
        """
        Abstract property to get the optimum solution of the problem.
        
        Returns:
            The optimum solution of the problem.
        """
        pass

    @property
    @abstractmethod
    def objs(self):
        """
        Abstract property to get the objectives of the problem.
        
        Returns:
            list: A list of objectives in the problem.
        """
        pass

    @property
    @abstractmethod
    def x(self):
        """
        Abstract property to get the decision variables of the problem.
        
        Returns:
            The decision variables of the problem.
        """
        pass

    @abstractmethod
    def optimize(self, *args):
        """
        Abstract method to perform optimization on the problem.

        Parameters:
            *args: Variable number of arguments to be used in the optimization process.

        Returns:
            The result of the optimization process.
        """
        pass


class w_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def w(self):
        """
        Abstract property to get the weights of the problem.
        
        Returns:
            The weights of the problem.
        """
        pass


class single_interface(metaclass=ABCMeta):
    # - propertys
    @property
    @abstractmethod
    def w(self):
        """
        Abstract property to get the weights of the problem.
        
        Returns:
            The weights of the problem.
        """
        pass

    def objetive(self):
        """
        Abstract method to define the objective to optimize.
        
        Returns:
            The objective to optimize.
        """
        pass


class box_interface(metaclass=ABCMeta):
    ## - propertys
    @property
    @abstractmethod
    def u(self):
        """
        Abstract property to get the upper bound of the problem.
        
        Returns:
            The upper bound of the problem.
        """
        pass

    @property
    @abstractmethod
    def l(self):
        """
        Abstract property to get the lower bound of the problem.
        
        Returns:
            The lower bound of the problem.
        """
        pass

    @property
    @abstractmethod
    def c(self):
        """
        Abstract property to get the constraints of the problem.
        
        Returns:
            The constraints of the problem.
        """
        pass
