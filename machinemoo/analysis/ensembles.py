import numpy as np
from typing import Any
from machinemoo.utils.typing import MatrixLike
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

def voting_ensemble(
    models: list[BaseEstimator | ClassifierMixin | RegressorMixin],
    X: MatrixLike,
    task: str = "classification",   # "classification" ou "regression"
    voting: str = "hard"            # "hard", "soft" (classificação) ou "mean" (regressão)
) -> Any:
    """Combine predictions from trained models into an ensemble result.
    
    Args:
        models (list): List of trained sklearn-like models.
        X (array-like): Input features for prediction.
        task (str, default="classification"): Task type - "classification" or "regression".
        voting (str, default="hard"): Voting strategy:
            - For classification: "hard" (majority voting) or "soft" (average probabilities).
            - For regression: only "mean" (average predictions).
    
    Returns
    
        Classification (hard): np.ndarray of predicted classes
        Classification (soft): (np.ndarray of predicted classes, np.ndarray of averaged probabilities)
        Regression (mean): np.ndarray of averaged predictions
    """
    
    if task == "classification":
        if voting == "hard":
            preds = np.array([model.predict(X) for model in models])
            # voto majoritário
            final_preds = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=preds
            )
            return final_preds
        
        elif voting == "soft":
            probs = np.array([model.predict_proba(X) for model in models])
            avg_probs = np.mean(probs, axis=0)
            final_preds = np.argmax(avg_probs, axis=1)
            return final_preds, avg_probs
        
        else:
            raise ValueError("For classification, voting must be 'hard' or 'soft'.")
    
    elif task == "regression":
        if voting == "mean":
            preds = np.array([model.predict(X) for model in models])
            final_preds = np.mean(preds, axis=0)
            return final_preds
        else:
            raise ValueError("For regression, only 'mean' voting is supported.")
    
    else:
        raise ValueError("Task must be either 'classification' or 'regression'.")

def weighted_soft_voting(
        models: list[BaseEstimator | ClassifierMixin | RegressorMixin],
        X: MatrixLike,
        weights: list[int | float]
    ):
    probs = np.array([model.predict_proba(X) for model in models])
    weighted_avg = np.average(probs, axis=0, weights=weights)
    preds = np.argmax(weighted_avg, axis=1)
    return preds, weighted_avg

def max_rule(models: list[BaseEstimator | ClassifierMixin | RegressorMixin], X: MatrixLike):
    probs = np.array([model.predict_proba(X) for model in models])
    max_probs = np.max(probs, axis=0)
    preds = np.argmax(max_probs, axis=1)
    return preds, max_probs