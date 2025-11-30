import math
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from sklearn.metrics import log_loss
from sklearn.utils.extmath import squared_norm
from sklearn.linear_model import LogisticRegression
from typing import Any
from numpy.typing import ArrayLike

from machinemoo import Scalarization
from machinemoo import (calculate_logreg_lipschitz_constant, 
                        calculate_l2_regularization_lipschitz_constant)
from machinemoo import get_logger
from machinemoo.utils.typing import MatrixLike

logger = get_logger(f"moo.{__name__}")

EPS = 1e-10

# Setting a seed for reproducibility
from machinemoo.utils.seed_config import set_np_torch_seed
set_np_torch_seed(42)

class TorchLogReg(nn.Module):
    def __init__(self, w_init, b_init) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=w_init.shape[0], out_features=1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(w_init).unsqueeze(0))
            self.linear.bias.copy_(torch.tensor([b_init]))
    
    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class LogRegScalarization(Scalarization):
    def __init__(
        self,
        num_objs: int,
        X: MatrixLike,
        y: ArrayLike,
        fair_feat: str,
        max_iter: int = 100,
        tol: float = 10**-4,
        lower_bound_estimate: str | float = 0.01, # Options: "zero", "lipschitz", or a float value
    ) -> None:
        super(LogRegScalarization, self).__init__(num_objs)
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = num_objs

        self.X = X
        self.y = y

        lambd = math.exp(-100)

        self.lower_bound_estimate = lower_bound_estimate
        if self.lower_bound_estimate == "lipschitz":
            self.L = np.zeros(num_objs)
            group_values = self.X[self.fair_feat].to_numpy()
            for g in self.fair_att:
                mask = (group_values == g)
                Xg = self.X.to_numpy()[mask]
                self.L[g] = calculate_logreg_lipschitz_constant(Xg)
            if self.M == 3:
                self.L[-1] = calculate_l2_regularization_lipschitz_constant()
        else:
            self.L = None
        self.__objs_lower = None  # Reset lower bound

        self.model = LogisticRegression(
            C=1 / lambd,
            tol=tol,
            solver="lbfgs",
            penalty="l2",
            max_iter=max_iter,
            warm_start=True,
            class_weight=None,
        )

    def _compute_lipschitz(self) -> np.ndarray:
        """Returns a lower estimative of the objective values with lipschitz estamation.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        J = np.array(self.gradient)
        grad_w = self.w@J
        L = self.w@self.L
        objs_delta = 1/L*J@grad_w - 1/2*self.L/(L**2)*(grad_w@grad_w)
        objs_lower = self.objs - objs_delta
        logger.debug("L", self.L)
        logger.debug("Objs delta:", objs_delta)
        logger.debug("Grad squar:", np.array([grad@grad for l, grad in zip(self.L, self.gradient)]))
        return objs_lower

    def get_gradient(
        self,
        X_train: MatrixLike,
        y_train: ArrayLike,
        model: Any,
        sample_weight_tensor = None
    ) -> npt.NDArray[np.float64]:
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32)

        w = model.coef_.flatten()       # shape: (n_features,)
        b = model.intercept_.item()

        pmodel = TorchLogReg(w, b)

        # Enable gradient tracking
        for param in pmodel.parameters():
            param.requires_grad = True

        criterion = nn.BCELoss(weight=sample_weight_tensor, reduction='mean')
        output = pmodel(X_tensor).squeeze()
        loss = criterion(output, y_tensor)
        loss.backward()

        grad_w = pmodel.linear.weight.grad.detach().numpy()
        grad_b = pmodel.linear.bias.grad.item()
        return np.concatenate([grad_w.flatten(), [grad_b]])

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[LogisticRegression, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[LogisticRegression, npt.NDArray[np.float64]]:
        
        if self.M == 2:
            fair_weight = weight
        elif self.M == 3:
            fair_weight = (weight[:-1]+EPS) / (1 - weight[-1] + EPS)
            self.model.set_params(C= (1 - weight[-1] + EPS) / (weight[-1] + EPS))
        else:
            print("Number of objective not supported in this scalarization!")

        # Compute fair weights
        fair_weights_dict = {
            ff: fw / sum(self.X[self.fair_feat] == ff)
            for ff, fw in zip(self.fair_att, fair_weight)
        }
        sample_weight = self.X[self.fair_feat].replace(fair_weights_dict)
        self.model.fit(self.X, self.y, sample_weight=sample_weight.to_numpy())
        y_pred = self.model.predict_proba(self.X)

        objs = np.zeros(self.M)
        for i, feat in enumerate(self.fair_att):
            fair_weight = np.zeros(len(self.fair_att))
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feat].replace(
                {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
            )
            objs[i] = log_loss(self.y, y_pred, sample_weight=sample_weight)

        # Set third objective as the squared L2 norm of the model coefficients
        if self.M == 3:
            objs[-1] = squared_norm(self.model.coef_)

        if self.lower_bound_estimate == "lipschitz":
            group_values = self.X[self.fair_feat].to_numpy()
            gradients = []
            for i, g in enumerate(self.fair_att):
                mask = (group_values == g)
                Xg = self.X.to_numpy()[mask]
                yg = self.y.to_numpy()[mask]

                fair_weight = np.zeros(len(self.fair_att))
                fair_weight[i] = 1
                sample_weight = self.X[self.fair_feat].replace(
                    {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
                )

                sample_weight_group = sample_weight[mask]
                sample_weight_tensor = torch.tensor(sample_weight_group.to_numpy(), dtype=torch.float32)

                grads = self.get_gradient(Xg, yg, self.model, sample_weight_tensor).squeeze()
                gradients.append(grads)

            if self.M == 3:
                grads = 2*np.concatenate([self.model.coef_.flatten(), [0]])
                gradients.append(grads)
            return self.model, objs, gradients
        return self.model, objs