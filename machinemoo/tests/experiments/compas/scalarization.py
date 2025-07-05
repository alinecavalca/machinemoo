import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils.extmath import squared_norm
from typing import Any
from numpy.typing import ArrayLike

from machinemoo import Scalarization
from machinemoo.utils.typing import MatrixLike

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class LogRegScalarization(Scalarization):
    def __init__(
        self,
        num_objs: int,
        X: MatrixLike,
        y: ArrayLike,
        fair_feat: str,
        max_iter: int = 100,
        tol: float = 10**-4,
        gradient:bool = False
    ) -> None:
        super(LogRegScalarization, self).__init__(num_objs)
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = num_objs

        self.X = X
        self.y = y

        self.use_gradient = gradient

        lambd = math.exp(-100)

        self.model = LogisticRegression(
            C=1 / lambd,
            tol=tol,
            solver="lbfgs",
            penalty="l2",
            max_iter=max_iter,
            warm_start=True,
            class_weight=None,
        )

    def get_gradient(
        self,
        X_train: MatrixLike,
        y_train: ArrayLike,
        model: Any
    ) -> npt.NDArray[np.float64]:
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
        
        w = torch.tensor(model.coef_, dtype=torch.float32, requires_grad=True)
        b = torch.tensor(model.intercept_, dtype=torch.float32)

        logits = X_tensor @ w.T + b  # shape: (n_samples, 1)
        preds = torch.sigmoid(logits)

        loss = torch.nn.functional.binary_cross_entropy(preds, y_tensor)

        loss.backward()

        return w.grad.detach().numpy().copy()

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[LogisticRegression, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[LogisticRegression, npt.NDArray[np.float64]]:
        if self.__M == 2:
            fair_weight = weight
        elif self.__M == 3:
            if weight[-1] == 0:
                lambd = 10**-20
            elif weight[-1] == 1:
                lambd = 10**20
            else:
                lambd = weight[-1] / (1 - weight[-1])

            fair_weight = weight[:-1] * (1 + lambd)
            self.model.C = 1 / lambd
        else:
            print("Number of objective not supported in this scalarization!")

        # sample_weight = self.X[self.fair_feat].replace({ff:fw/sum(self.X[self.fair_feat]==ff) for ff, fw in zip(self.fair_att,fair_weight)}) + 10**-20
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

        if self.__M == 3:
            objs[-1] = squared_norm(self.model.coef_)

        if self.use_gradient is True:
            #gradient = self.get_gradient(self.X, self.y, self.model).squeeze()
            #gradient = np.vstack(gradient)
            group_values = self.X[self.fair_feat].to_numpy()

            grads = []
            for g in self.fair_att:
                mask = (group_values == g)
                Xg = self.X.to_numpy()[mask]
                yg = self.y.to_numpy()[mask]

                grad = self.get_gradient(Xg, yg, self.model)
                grads.append(grad)

            grads = np.vstack(grads) 
            print(f"gradi  {grads}")
            return self.model, objs, grads
        return self.model, objs


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
            # nn.Linear(hidden_dim, 128),
            # nn.ReLU(),
            # nn.Linear(128, hidden_dim),
            # nn.ReLU(),
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
        num_objs: int = 2,
        gradient: bool = False
    ) -> None:
        super(MLPScalarization, self).__init__(num_objs)
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = num_objs

        self.N = X.shape[0]
        self.X = X
        self.y = y

        self.use_gradient = gradient

        self.X_tensor = torch.tensor(self.X.to_numpy(), dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y.to_numpy(), dtype=torch.float32)

        input_dim = self.X.shape[1]
        hidden_dim = 64
        output_dim = 1
        self.learning_rate = 0.01
        # self.learning_rate = 0.1
        self.num_epochs = 200

        self.model = MLP(input_dim, hidden_dim, output_dim)

    def training(
        self,
        weight:npt.NDArray[np.float64]
    ) -> tuple[MLP, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[MLP, npt.NDArray[np.float64]]:
        fair_weight = weight

        fair_weights_dict = {
            ff: fw / sum(self.X[self.fair_feat] == ff)
            for ff, fw in zip(self.fair_att, fair_weight)
        }
        sample_weight = self.X[self.fair_feat].replace(fair_weights_dict)  # +10**-20

        self.sample_weight = torch.tensor(sample_weight.to_numpy(), dtype=torch.float32)

        criterion = nn.BCELoss(weight=self.sample_weight, reduction="sum")
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
            if self.use_gradient is True:
                mask = self.X[self.fair_feat] == feat
                X_group = self.X_tensor[mask.to_numpy()]
                y_group = self.y_tensor[mask.to_numpy()]

                self.model.zero_grad()

                # Enable gradient computation
                X_group.requires_grad = True

                y_pred_group = self.model(X_group).squeeze()
                loss_group = nn.BCELoss(reduction="sum")(y_pred_group, y_group)
                loss_group.backward()

                # Get gradients from the last layer
                grad_w = self.model.layers[-2].weight.grad.detach().numpy().flatten()
                grad_b = self.model.layers[-2].bias.grad.detach().numpy().flatten()
                grad_vec = np.concatenate([grad_b, grad_w])
                gradients.append(grad_vec)

            fair_weight = np.zeros(self.M)
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feat].replace(
                {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
            )
            objs[i] = log_loss(
                self.y, y_pred.detach().numpy(), sample_weight=sample_weight
            )

        if self.use_gradient is True:
            # Stack gradients across all objectives
            gradient = np.vstack(gradients)
            return self.model, objs, gradient
        return self.model, objs
