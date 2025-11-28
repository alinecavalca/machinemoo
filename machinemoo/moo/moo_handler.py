import numpy as np
import numpy.typing as npt
from typing import Any, Literal
from collections.abc import Callable
from inspect import Parameter, signature

# Import only the refactored algorithm
from machinemoo.moo.random_weights import RandomWeights
from machinemoo.utils.logging_config import get_logger
from machinemoo.utils.typing import scalar

__all__ = [
    'MachineMoo',
    'get_objectives',
    'get_models',
    'run_ensemble'
]

class MachineMoo:
    """
    Machine Learning Multi-Objective Optimization Handler.
    
    Acts as a facade for executing various MOO algorithms.
    Currently refactored to support: RandomWeights.
    """

    def __init__(
        self,
        weighted_scalar: scalar,
        single_scalar: scalar | None = None,
        verbose: bool = False,
        debug: bool = False
    ) -> None:
        """
        Initializes the handler.

        Args:
            weighted_scalar (scalar): Strategy for weighted sums.
            single_scalar (scalar, optional): Strategy for single objectives.
            verbose (bool): Enable info logging.
            debug (bool): Enable debug logging.
        """
        self.logger = get_logger(name='moo', verbose=verbose, debug=debug)
        self._weighted_scalar = weighted_scalar
        self._single_scalar = single_scalar if single_scalar else weighted_scalar
        self._verbose = verbose
        self._debug = debug

    def _filter_kwargs(
        self,
        fn: Callable,
        params: dict[str, Any],
        to_camel: bool = False
    ) -> dict[str, Any]:
        """
        Filters kwargs to match the target function's signature.
        """
        sig = signature(fn)
        valid_keys = {
            k for k, p in sig.parameters.items()
            if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        }

        if to_camel:
            converted_params = {self._snake_to_camel(k): v for k, v in params.items()}
        else:
            converted_params = params

        filtered = {k: v for k, v in converted_params.items() if k in valid_keys}
        return filtered

    def _snake_to_camel(self, snake_str: str) -> str:
        """Converts snake_case to camelCase (legacy support)."""
        parts = snake_str.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    def mo_optimization(self, method: str, **kwargs: Any) -> Any:
        """
        Executes the specified MOO method.
        
        Args:
            method (str): Name of the method ('random_weights').
            **kwargs: Additional arguments for the optimizer.
        """
        method = method.lower()
        
        if method == 'random_weights':
            return self.moo_random_weights(**kwargs)
        
        # Placeholders for pending refactoring
        # elif method == 'monise':
        #     return self.moo_monise(**kwargs)
        # elif method == 'mola':
        #     return self.moo_mola(**kwargs)
        # elif method == 'nise':
        #     return self.moo_nise(**kwargs)
        
        else:
            raise ValueError(f"Unknown or currently unsupported method: {method}")

    def moo_random_weights(self, **kwargs: Any) -> RandomWeights:
        """Runs the Random Weights optimization."""
        # The refactored RandomWeights uses snake_case, so to_camel=False
        filtered = self._filter_kwargs(RandomWeights, params=kwargs, to_camel=False)
        
        optimizer = RandomWeights(
            weighted_scalar=self._weighted_scalar,
            single_scalar=self._single_scalar,
            verbose=self._verbose,
            debug=self._debug,
            **filtered
        )
        optimizer.optimize()
        return optimizer


def get_objectives(optimizer: Any) -> npt.NDArray[np.float64]:
    """
    Extracts objective vectors from an optimizer's solution list.
    Supports both refactored (solutions_list) and legacy (solutionsList) attributes.
    """
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.objs for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.objs for s in optimizer.solutionsList])
        
    raise ValueError("Optimizer does not contain a valid solutions list attribute.")


def get_models(optimizer: Any) -> Any:
    """
    Extracts trained models from an optimizer's solution list.
    Supports both refactored (solutions_list) and legacy (solutionsList) attributes.
    """
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.x for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.x for s in optimizer.solutionsList])
        
    raise ValueError("Optimizer does not contain a valid solutions list attribute.")


def run_ensemble(
    optimizer: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    models: Any = None,
    ensemble_type: str = 'voting',
    voting_type: Literal['hard', 'soft'] = 'soft'
) -> float:
    """
    Runs an ensemble model on the solutions produced by a MOO optimizer.
    (Implementation placeholder matching previous version).
    """
    # Logic for running ensemble would go here
    # Placeholder implementation to satisfy the declared return type.
    # TODO: implement ensemble evaluation and return a relevant metric (e.g., accuracy or score).
    return 0.0