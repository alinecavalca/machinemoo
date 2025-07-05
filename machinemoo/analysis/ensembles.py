import numpy as np
import numpy.typing as npt
from typing import Any, Literal
from sklearn.exceptions import NotFittedError
from sklearn.ensemble._bagging import BaggingRegressor
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, VotingClassifier)

import pandas as pd
from numpy.typing import ArrayLike

from machinemoo.utils.typing import MatrixLike
from machinemoo.utils.logging_config import logger

class Ensemble():
    """
    Wrapper for ensemble learning using multiple trained models obtained 
    through multi-objective optimization.

    Currently supports scikit-learn ensembles such as VotingClassifier, 
    BaggingClassifier, and AdaBoostClassifier.
    """
    def __init__(
            self, 
            models: Any, 
            X_train: MatrixLike | ArrayLike, 
            y_train: ArrayLike,
            ensemble_type: str = 'voting', 
            voting_type: Literal['hard', 'soft'] = 'soft',
        ) -> None:
        """Initializes the Ensemble object.

        Args:
            models (Any): List of trained models to include in the ensemble.
            X_train (ArrayLike): Training features.
            y_train (ArrayLike): Training labels.
            ensemble_type (str, optional): Type of ensemble method to use. Defaults to 'voting'.
            voting_type (str, optional): Voting strategy for classification ('hard' or 'soft'). Defaults to 'soft'.
        """
        self.ensemble_type  = ensemble_type
        self.voting_type = voting_type

        self.models  = models
        self.X_train = X_train
        self.y_train = y_train
        self.ensemble_model= self._create_ensemble()

    def _create_ensemble(self) -> Any:
        """Creates and fits the ensemble model based on the selected type.

        Supported types:
            - 'voting': Uses VotingClassifier with the provided models.
            - 'baggin': Uses BaggingClassifier (with the first model as base).
            - 'adaboost': Uses AdaBoostClassifier (with the first model as base).

        Returns:
            Any: A fitted ensemble model.
        """
        ensemble_model = None
        try:
            if self.ensemble_type == 'voting':
                ensemble_model = VotingClassifier(estimators=[(f'model_{i}', model) for i, model in enumerate(self.models)],
                                                    voting=self.voting_type)
            elif self.ensemble_type == 'baggin':
                ensemble_model = BaggingClassifier(estimator=self.models[0], n_estimators=10)

            elif self.ensemble_type == 'adaboost':
                ensemble_model = AdaBoostClassifier(estimator=self.models[0], n_estimators=10)
            else:
                raise ValueError(f"Unsupported ensemble type: '{self.ensemble_type}'")

            return ensemble_model.fit(self.X_train, self.y_train)

        except Exception as e:
            logger.error(f"Failed to create ensemble model ({self.ensemble_type}): {e}")
            return None

    def predict(self, X: npt.NDArray[np.float64]) -> Any | None:
        """Makes predictions using the ensemble model.

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            Any or None: Predicted labels, or None if prediction fails.
        """
        try:
            return self.ensemble_model.predict(X)
        except NotFittedError:
            logger.warning("Ensemble model has not been fitted.")
        except ValueError as e:
            logger.warning(f"Invalid input data: {e}")
        except AttributeError as e:
            logger.warning(f"Ensemble model is not properly initialized: {e}")
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
        return None

    def predict_proba(self, X: npt.NDArray[np.float64]) -> Any | None:
        """Predicts class probabilities using the ensemble model.

        Only available if the ensemble supports `predict_proba` (e.g., soft voting).

        Args:
            X (np.ndarray): Input feature data.

        Returns:
            Any or None: Predicted probabilities, or None if prediction fails.
        """
        try:
            return self.ensemble_model.predict_proba(X)
        except NotFittedError:
            logger.warning("Ensemble model has not been fitted.")
        except ValueError as e:
            logger.warning(f"Invalid input data: {e}")
        except AttributeError as e:
            logger.warning(f"Ensemble model is not properly initialized: {e}")
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
        return None