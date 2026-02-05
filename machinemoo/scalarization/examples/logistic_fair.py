import math
import numpy as np
import numpy.typing as npt
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import squared_norm

# Updated imports: Scalarization -> BaseScalarizer
from machinemoo.scalarization.core import BaseScalarizer, LipschitzRegLoghMixin
from machinemoo.scalarization.lipschitz_estimation import (
    calculate_logreg_lipschitz_constant, 
    calculate_l2_regularization_lipschitz_constant
)
from machinemoo.utils.torch_helpers import get_logistic_gradient
from machinemoo.utils.typing import MatrixLike
from machinemoo.utils import get_logger

ArrayLike = npt.ArrayLike

logger = get_logger(f"moo.{__name__}")
EPS = 1e-10

class FairLogRegMO(LipschitzRegLoghMixin, BaseScalarizer):
    """
    Scalarization for Logistic Regression with Fairness objectives.
    
    Inherits from LipschitzRegLoghMixin to automatically handle the 
    Lipschitz-based lower bound estimation logic.
    """
    def __init__(
        self,
        num_objs: int,
        X: MatrixLike,
        y: ArrayLike,
        fair_feat: str,
        max_iter: int = 100,
        tol: float = 10**-4,
        lower_bound_estimate: str | float = "zero",
    ) -> None:
        """
        Args:
            num_objs (int): Number of objectives (2 or 3).
            X, y: Training data.
            fair_feat (str): Name of the column in X representing the sensitive attribute.
            lower_bound_estimate: "zero", "lipschitz", or a float factor.
        """
        # Initialize base class
        super().__init__(num_objs=num_objs, lower_bound_estimate=lower_bound_estimate)
        
        self.fair_feat = fair_feat
        # Use numpy's unique which works for both pandas Series and numpy arrays
        self.fair_att = sorted(np.unique(X[fair_feat]))
        
        self.X = X
        self.y = y

        # Calculate Lipschitz Constants if required
        if self.lower_bound_estimate == "lipschitz":
            self.L = np.zeros(num_objs)
            group_values = np.array(self.X[self.fair_feat])
            
            for g in self.fair_att:
                mask = (group_values == g)
                Xg = np.array(self.X)[mask]
                
                idx = int(g) if isinstance(g, (int, float, np.number)) else self.fair_att.index(g)
                if idx < num_objs:
                    self.L[idx] = calculate_logreg_lipschitz_constant(Xg)
            
            if self.M == 3:
                self.L[-1] = calculate_l2_regularization_lipschitz_constant()
        else:
            self.L = None

        # Initialize Model
        lambd = math.exp(-100)
        self.model = LogisticRegression(
            C=1 / lambd,
            tol=tol,
            solver="lbfgs",
            penalty="l2",
            max_iter=max_iter,
            warm_start=True
        )

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[LogisticRegression, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[LogisticRegression, npt.NDArray[np.float64]]:
        
        # 1. Adjust weights for regularization (if 3 objectives)
        if self.M == 2:
            fair_weight = weight
        elif self.M == 3:
            fair_weight = (weight[:-1] + EPS) / (1 - weight[-1] + EPS)
            # Update regularization C based on the 3rd weight
            C_new = (1 - weight[-1] + EPS) / (weight[-1] + EPS)
            self.model.set_params(C=C_new)
        else:
            logger.warning("Number of objectives not fully supported logic in training.")
            fair_weight = weight

        # 2. Compute sample weights for fairness
        fair_weights_dict = {
            ff: fw / max(1, sum(self.X[self.fair_feat] == ff)) # Avoid div/0
            for ff, fw in zip(self.fair_att, fair_weight)
        }
        sample_weight = np.array([fair_weights_dict[ff] for ff in self.X[self.fair_feat]])
        
        # 3. Train Model
        self.model.fit(self.X, self.y, sample_weight=sample_weight)
        y_pred = self.model.predict_proba(self.X)

        # 4. Calculate Objectives
        objs = np.zeros(self.M)
        for i, feat in enumerate(self.fair_att):
            if i >= self.M:
                break
            
            mask = np.asarray(self.X[self.fair_feat] == feat)
            if mask.sum() > 0:
                y_arr = np.asarray(self.y)
                y_pred_arr = np.asarray(y_pred)
                objs[i] = log_loss(y_arr[mask], y_pred_arr[mask])

        # Objective 3: L2 norm of coefficients
        if self.M == 3:
            objs[-1] = squared_norm(self.model.coef_)

        # 5. Calculate Gradients (Only if Lipschitz estimation is active)
        if self.lower_bound_estimate == "lipschitz":
            group_values = np.array(self.X[self.fair_feat])
            gradients = []
            
            for i, g in enumerate(self.fair_att):
                if i >= self.M and self.M != 3: 
                    break

                mask = (group_values == g)
                Xg = np.array(self.X)[mask]
                yg = np.array(self.y)[mask]

                if len(Xg) > 0:
                    grads = get_logistic_gradient(Xg, yg, self.model, sample_weight_tensor=None).squeeze()
                    gradients.append(grads)
                else:
                    gradients.append(np.zeros(self.model.coef_.size + 1))

            if self.M == 3:
                grads = 2 * np.concatenate([self.model.coef_.flatten(), [0]])
                gradients.append(grads)
            
            gradients_arr = np.asarray(gradients, dtype=np.float64)
            return self.model, objs, gradients_arr

        return self.model, objs