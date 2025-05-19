import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


class Ensemble():
    def __init__(self, models, solutions_list, ensemble_type='voting', voting_type='soft',
                  X_train=None, y_train=None):
        # self.solutions = solutions
    
        self.solutions_list = solutions_list
        self.ensemble_type  = ensemble_type
        self.voting_type    = voting_type
        self.models         = models  # Aqui pegamos o modelo ajustado de cada solução
        self.X_train        = X_train
        self.y_train        = y_train
        self.ensemble_model = self._create_ensemble()


    def _create_ensemble(self):
        """
        Creates the ensemble model using the specified ensemble type.

        Returns:
            ensemble_model: The ensemble model (VotingClassifier, etc.).
        """
        if self.ensemble_type == 'voting':
            ensemble_model = VotingClassifier(estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                                                voting=self.voting_type).fit(self.X_train, self.y_train)
        elif self.ensemble_type == 'baggin':
            ensemble_model = BaggingClassifier(estimator=self.models[0], n_estimators=10).fit(self.X_train, self.y_train)

        elif self.ensemble_type == 'adaboost':
            ensemble_model = AdaBoostClassifier(estimator=self.models[0], n_estimators=10).fit(self.X_train, self.y_train)

        return ensemble_model

    # def fit(self, X, y):
    #     """
    #     Fit the ensemble model to the data.

    #     Parameters:
    #         X (ndarray): The feature data.
    #         y (ndarray): The target labels.

    #     Returns:
    #         None
    #     """
    #     self.ensemble_model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the ensemble model.

        Parameters:
            X (ndarray): The feature data.

        Returns:
            ndarray: The predicted labels.
        """
        return self.ensemble_model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the ensemble model (only available for certain models).

        Parameters:
            X (ndarray): The feature data.

        Returns:
            ndarray: The predicted probabilities.
        """
        return self.ensemble_model.predict_proba(X)