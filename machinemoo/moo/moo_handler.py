import pandas as pd
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike 
from typing import Any, Literal
from collections.abc import Callable
from inspect import Parameter, signature
from sklearn.metrics import accuracy_score

from machinemoo.moo.nise import nise
from machinemoo.moo.mola import Mola
from machinemoo.moo.monise import monise
from machinemoo.moo.rennen import rennen
from machinemoo.moo.random_weights import random_weights
from machinemoo.analysis.ensembles import Ensemble
from machinemoo.utils.logging_config import logger
from machinemoo.utils.typing import scalar, MatrixLike

__all__ = [
    'MachineMoo',
    'get_objectives',
    'run_ensemble'
]

class MachineMoo:
    """Machine Learning Multi-Objective Optimization Handler.

    This class handles the selection and execution of various
    multi-objective optimization (MOO) methods based on scalarization
    techniques. It provides a unified interface to run and manage 
    different optimizers such as MOLA, MONISE, and Random Weights.
    """

    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar | None = None,
    ) -> None:
        """"Initializes the optimization handler with scalarization strategies.

        Args:
            weighted_scalar (scalar): Scalarization method used for weighted objectives.
            single_scalar (scalar, optional): Scalarization method for single-objective
                problems. Defaults to the same as `weighted_scalar` if not provided.
        """
        self._weighted_scalar: scalar = weighted_scalar
        #self._single_scalar = single_scalar or weighted_scalar
        self._single_scalar: scalar | None = single_scalar

        if self._single_scalar == None:
            self._single_scalar = self._weighted_scalar

    def _filter_kwargs(
        self,
        fn: Callable,
        params: dict[str, Any],
        to_camel: bool = False
    ) -> dict[str, Any]:
        """Filters a dictionary of parameters, keeping only those valid for a given function.

        Optionally converts snake_case to camelCase to match API expectations.

        Args:
            fn (Callable): The target function or class to match parameters for.
            params (dict[str, Any]): The input parameter dictionary.
            to_camel (bool, optional): Whether to convert keys to camelCase. Default is False.

        Returns:
            dict[str, Any]: Filtered dictionary with only valid parameters.
        """
        sig = signature(fn)
        valid_keys = {
            k for k, p in sig.parameters.items()
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

        if to_camel:
            # Converte todas as chaves snake_case -> camelCase
            converted_params = {self.snake_to_camel(k): v for k, v in params.items()}
        else:
            converted_params = params

        filtered = {k: v for k, v in converted_params.items() if k in valid_keys}
        ignored = set(converted_params) - valid_keys

        if ignored:
            logger.warning(f"Ignored parameters for {fn.__name__}: {ignored}")
        logger.debug(f"Filtered parameters for {fn.__name__}: {filtered}")
        return filtered

    
    def snake_to_camel(self, snake_str: str) -> str:
        """Converts a snake_case string to camelCase.

        Args:
            snake_str (str): String in snake_case format.

        Returns:
            str: Converted string in camelCase.
        """
        parts = snake_str.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    

    def mo_optimization(self, method: str, **kwargs: Any) -> Any:
        """Selects and performs a multi-objective optimization method.

        Args:
            method (str): Name of the optimization method to use. Must be one of:
                'mola', 'monise', 'random_weight'.

        Returns:
            Any: The optimizer instance after execution.

        Raises:
            ValueError: If the specified method is not supported.
        """
        methods = {
            'mola': self.moo_mola,
            'monise': self.moo_monise,
            'random_weight': self.moo_random_weight,
            #'rennen': self.moo_rennen,
            #'nise': self.moo_nise
        }

        if method not in methods:
            raise ValueError(f"Unknown optimization method: {method}")

        optimizer = methods[method](**kwargs)
        return optimizer

    def moo_mola(self, **kwargs: Any) -> Any:
        """Runs the MOLA optimization algorithm.

        Returns:
            Any: The MOLA optimizer instance after running optimization.
        """
        filtered = self._filter_kwargs(Mola, params=kwargs)
        optimizer = Mola(
            weighted_scalar=self._weighted_scalar,
            single_scalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer
    
    def moo_monise(self, **kwargs: Any) -> Any:
        """Runs the MONISE optimization algorithm.

        Returns:
            Any: The MONISE optimizer instance after running optimization.
        """
        filtered = self._filter_kwargs(monise, params=kwargs, to_camel=True)
        optimizer = monise(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_nise(self, **kwargs: Any) -> Any:
        """Runs the NISE optimization algorithm.

        Returns:
            Any: The NISE optimizer instance after running optimization.
        """
        filtered = self._filter_kwargs(nise, params=kwargs, to_camel=True)
        optimizer = nise(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_random_weight(self, **kwargs: Any) -> Any:
        """Runs the Random Weights optimization algorithm.

        Returns:
            Any: The Random Weights optimizer instance after running optimization.
        """
        filtered = self._filter_kwargs(random_weights, params=kwargs, to_camel=True)
        optimizer = random_weights(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_rennen(self, **kwargs: Any) -> Any:
        """Runs the Rennen optimization algorithm.

        Returns:
            Any: The Rennen optimizer instance after running optimization.
        """
        filtered = self._filter_kwargs(random_weights, params=kwargs, to_camel=True)
        optimizer = rennen(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

def get_objectives(optimizer: Any) -> npt.NDArray[np.float64]:
    """Extracts objective vectors from an optimizer's solution list.

    Supports both `solutions_list` and `solutionsList` naming styles.

    Args:
        optimizer (Any): The optimizer instance.

    Returns:
        np.ndarray: Array of objective vectors.

    Raises:
        ValueError: If no recognizable solution list is found.
    """
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.objs for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.objs for s in optimizer.solutionsList])
    raise ValueError("Optimizer does not contain recognizable solution list attribute")

def get_models(optimizer: Any) -> Any:
    """Extracts model objects (x) from an optimizer's solution list.

    Supports both `solutions_list` and `solutionsList` naming styles.

    Args:
        optimizer (Any): The optimizer instance.

    Returns:
        Any: List or array of model objects (`x` values).

    Raises:
        ValueError: If no recognizable solution list is found.
    """
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.x for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.x for s in optimizer.solutionsList])
    raise ValueError("Optimizer does not contain recognizable solution list attribute")

def run_ensemble(
    optimizer: Any,
    X_train: MatrixLike | ArrayLike,
    y_train: ArrayLike,
    X_test: MatrixLike | ArrayLike,
    y_test: ArrayLike,
    models: Any = None,
    ensemble_type: str = 'voting',
    voting_type: Literal['hard', 'soft'] = 'soft'
) -> float:
    """Runs an ensemble model on the solutions produced by a MOO optimizer.

    Uses the stored models from the optimizer (or explicitly provided ones)
    to perform ensemble prediction and compute the test accuracy.

    Args:
        optimizer (Any): The MOO optimizer instance.
        X_train (array-like): Training feature matrix.
        y_train (array-like): Training labels.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Test labels.
        models (Any, optional): Pretrained models to use in the ensemble. Defaults to None.
        ensemble_type (str, optional): Type of ensemble method ('voting', etc.). Defaults to 'voting'.
        voting_type (str, optional): Voting strategy ('soft' or 'hard'). Defaults to 'soft'.

    Returns:
        float: Accuracy of the ensemble model on the test set.

    Raises:
        ValueError: If models could not be extracted or are missing.
    """
    if models is None:
        models = getattr(optimizer, 'get_models', lambda: None)()
    if models is None:
        raise ValueError("optimizer does not support model extraction for ensemble.")

    ensemble_model = Ensemble(
        models=models,
        ensemble_type=ensemble_type,
        voting_type=voting_type,
        X_train=X_train,
        y_train=y_train
    )

    predictions = ensemble_model.predict(X_test)
    accuracy: float = accuracy_score(y_test, predictions)
    #print(f'Ensemble Accuracy: {accuracy:.4f}')
    return accuracy