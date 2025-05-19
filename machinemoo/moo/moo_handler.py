import numpy as np
from typing import Optional, Union, Dict, Any, Callable

from inspect import signature, Parameter

from machinemoo.moo import scalar_interface, w_interface, single_interface
from machinemoo.moo.nise import nise
from machinemoo.moo.mola import Mola
from machinemoo.moo.monise import monise
from machinemoo.moo.rennen import rennen
from machinemoo.moo.random_weights import random_weights
from machinemoo.analysis.ensembles import Ensemble
from machinemoo.utils.logging_config import logger

from sklearn.metrics import accuracy_score

__all__ = [
    'MLMoo',
    'get_objectives',
    'run_ensemble'
]

class MLMoo:
    """
    Machine Learning Multi-Objective Optimization Handler.
    This class handles the selection and execution of various multi-objective optimization methods
    using scalarization interfaces.
    """

    def __init__(
        self,
        weighted_scalar: Union[scalar_interface, w_interface],
        single_scalar: Optional[Union[scalar_interface, single_interface]] = None,
    ) -> None:
        """
        Initialize the optimization handler with scalarization methods.

        Parameters
        ----------
        weighted_scalar : scalar_interface
            Scalarization method for weighted objectives.
        single_scalar : scalar_interface, optional
            Scalarization method for single-objective, defaults to weighted_scalar.
        """
        self._weighted_scalar = weighted_scalar
        #self._single_scalar = single_scalar or weighted_scalar
        self._single_scalar = single_scalar

        if self._single_scalar == None:
            self._single_scalar = self._weighted_scalar

    def _filter_kwargs(
        self,
        fn: Callable,
        params: Dict[str, Any],
        to_camel: bool = False
    ) -> Dict[str, Any]:
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
        parts = snake_str.split('_')
        return parts[0] + ''.join(word.capitalize() for word in parts[1:])

    

    def mo_optimization(self, method: str, **kwargs) -> Any:
        """Selects and performs a multi-objective optimization method with filtered parameters."""
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

    def moo_mola(self, **kwargs) -> Any:
        filtered = self._filter_kwargs(Mola, params=kwargs)
        optimizer = Mola(
            weighted_scalar=self._weighted_scalar,
            single_scalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer
    
    def moo_monise(self, **kwargs) -> Any:
        print("monise")
        filtered = self._filter_kwargs(monise, params=kwargs, to_camel=True)
        optimizer = monise(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_nise(self, **kwargs) -> Any:
        filtered = self._filter_kwargs(nise, params=kwargs, to_camel=True)
        optimizer = nise(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_random_weight(self, **kwargs) -> Any:
        filtered = self._filter_kwargs(random_weights, params=kwargs, to_camel=True)
        optimizer = random_weights(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer

    def moo_rennen(self, **kwargs) -> Any:
        filtered = self._filter_kwargs(random_weights, params=kwargs, to_camel=True)
        optimizer = rennen(
            weightedScalar=self._weighted_scalar,
            singleScalar=self._single_scalar,
            **filtered
        )
        optimizer.optimize()
        return optimizer
        

    #@staticmethod
def get_objectives(optimizer: Any) -> np.ndarray:
    """Extracts objective vectors from a optimizer's solution list."""
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.objs for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.objs for s in optimizer.solutionsList])
    raise ValueError("optimizer does not contain recognizable solution list attribute")

def get_models(optimizer: Any) -> np.ndarray:
    """Extracts objective vectors from a optimizer's solution list."""
    if hasattr(optimizer, 'solutions_list'):
        return np.array([s.x for s in optimizer.solutions_list])
    elif hasattr(optimizer, 'solutionsList'):
        return np.array([s.x for s in optimizer.solutionsList])
    raise ValueError("optimizer does not contain recognizable solution list attribute")

def run_ensemble(
    optimizer: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    models: Any = None,
    ensemble_type: str = 'voting',
    voting_type: str = 'soft'
) -> float:
    """
    Run an ensemble model on the Pareto solutions, using stored solutions.

    Parameters
    ----------
    models : list
        List of trained ML models.
    sol : list
        Corresponding solutions (e.g., objective vectors).
    X_train, y_train, X_test, y_test : array-like
        Training and testing data.
    ensemble_type : str
        Type of ensemble. Default is 'voting'.
    voting_type : str
        Type of voting: 'soft' or 'hard'.

    Returns
    -------
    float
        Accuracy score on the test set.
    """
    if models is None:
        models = getattr(optimizer, 'get_models', lambda: None)()
    if models is None:
        raise ValueError("optimizer does not support model extraction for ensemble.")

    ensemble_model = Ensemble(
        models,
        optimizer,
        ensemble_type=ensemble_type,
        voting_type=voting_type,
        X_train=X_train,
        y_train=y_train
    )

    predictions = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    #print(f'Ensemble Accuracy: {accuracy:.4f}')
    return accuracy