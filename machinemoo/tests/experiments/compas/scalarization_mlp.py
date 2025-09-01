import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt
from sklearn.metrics import log_loss
from numpy.typing import ArrayLike

from machinemoo import get_logger
from machinemoo.scalarization.moo_scalarization import Scalarization
from machinemoo import calculate_torch_lipschitz_constant
from machinemoo.utils.typing import MatrixLike

# Setting a seed for reproducibility
from machinemoo.utils.seed_config import set_np_torch_seed
set_np_torch_seed(42)

logger = get_logger(f"moo.{__name__}")

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class MLPScalarization(Scalarization):
    def __init__(
        self,
        X: MatrixLike,
        y: ArrayLike,
        fair_feat: str,
        epochs: int = 200,
        num_objs: int = 2,
        lower_bound_estimate: str | float = 0.01, # Options: "zero", "lipschitz", or a float value
    ) -> None:
        super(MLPScalarization, self).__init__(num_objs)
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = num_objs

        self.N = X.shape[0]
        self.X = X
        self.y = y

        self.X_tensor = torch.tensor(self.X.to_numpy(), dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y.to_numpy(), dtype=torch.float32)

        input_dim = self.X.shape[1]
        hidden_dim = 64
        output_dim = 1
        self.learning_rate = 0.01
        self.num_epochs = epochs

        self.model = MLP(input_dim, hidden_dim, output_dim)

        self.lower_bound_estimate = lower_bound_estimate

        # Handle lower_bound_estimate option
        if lower_bound_estimate == "lipschitz":
            self.L = np.zeros(len(self.fair_att))
            group_values = self.X[self.fair_feat].to_numpy()
            for i, g in enumerate(self.fair_att):
                mask = (group_values == g)
                Xg = torch.tensor(self.X.to_numpy()[mask], dtype=torch.float32)
                yg = torch.tensor(self.y.to_numpy()[mask], dtype=torch.float32)

                fair_weight = np.zeros(self.M)
                fair_weight[i] = 1
                sample_weight = self.X[self.fair_feat].replace(
                    {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
                )

                sample_weight_group = sample_weight[mask]
                sample_weight_tensor = torch.tensor(sample_weight_group.to_numpy(), dtype=torch.float32)
                if sample_weight_tensor.ndim > 1:
                    sample_weight_tensor = sample_weight_tensor.squeeze()

                lip_criterion = nn.BCELoss(weight=sample_weight_tensor, reduction="mean") 
                
                loss_fn = lambda: lip_criterion(self.model(Xg).squeeze(), yg)
                self.L[g] = calculate_torch_lipschitz_constant(
                        model=self.model,
                        loss_fn=loss_fn,
                        num_iterations=20,
                    )

            logger.debug("Lipschitz constants for each group:", self.L)
        else:
            self.L = None

    def _compute_lipschitz(self) -> np.ndarray:
        """Returns a lower estimative of the objective values with lipschitz estamation.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        w_gradient = self.w@np.array([self.gradient[idx] for idx in range(self.M)])
        objs_delta = 1/(2*self.w@self.L)*w_gradient@w_gradient
        objs_lower = self.objs - objs_delta
        return objs_lower

    def training(
        self,
        weight:npt.NDArray[np.float64]
    ) -> tuple[MLP, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[MLP, npt.NDArray[np.float64]]:
        fair_weight = weight

        fair_weights_dict = {
            ff: fw / sum(self.X[self.fair_feat] == ff)
            for ff, fw in zip(self.fair_att, fair_weight)
        }
        sample_weight = self.X[self.fair_feat].replace(fair_weights_dict)

        self.sample_weight = torch.tensor(sample_weight.to_numpy(), dtype=torch.float32)

        criterion = nn.BCELoss(weight=self.sample_weight, reduction="mean")
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )

        self.model.train()
        for epoch in range(self.num_epochs):
            y_pred = self.model(self.X_tensor).squeeze()
            loss = criterion(y_pred, self.y_tensor)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_tensor)

        objs = np.zeros(self.M)
        gradients = []

        for i, feat in enumerate(self.fair_att):
            fair_weight = np.zeros(self.M)
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feat].replace(
                {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
            )
            if self.lower_bound_estimate == "lipschitz":
                mask = self.X[self.fair_feat] == feat
                X_group = self.X_tensor[mask.to_numpy()]
                y_group = self.y_tensor[mask.to_numpy()]

                self.model.zero_grad()

                # Enable gradient computation
                X_group.requires_grad = True

                y_pred_group = self.model(X_group).squeeze()

                sample_weight_group = sample_weight[mask]
                sample_weight_tensor = torch.tensor(sample_weight_group.to_numpy(), dtype=torch.float32)
                if sample_weight_tensor.ndim > 1:
                    sample_weight_tensor = sample_weight_tensor.squeeze()

                loss_group = nn.BCELoss(
                    weight=sample_weight_tensor, reduction="mean"
                )(y_pred_group, y_group)
                #loss_group = nn.BCELoss(weight=sample_weight_tensor, reduction="sum")(y_pred_group, y_group)
                loss_group.backward()

                # Get gradients of the model parameters
                grad_vec = np.concatenate([
                    param.grad.view(-1).detach().numpy() for param in self.model.parameters()
                ])
                gradients.append(grad_vec)

            
            objs[i] = log_loss(
                self.y, y_pred.detach().numpy(), sample_weight=sample_weight
            )

        if self.lower_bound_estimate == "lipschitz":
            # Stack gradients across all objectives
            gradient = np.vstack(gradients)
            return self.model, objs, gradient
        return self.model, objs
