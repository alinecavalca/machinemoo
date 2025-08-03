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
from scipy.special import expit
from tqdm.auto import tqdm

EPS = 1e-10

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class TorchLogReg(nn.Module):
    def __init__(self, w_init, b_init):
        super().__init__()
        self.linear = nn.Linear(in_features=w_init.shape[0], out_features=1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(w_init).unsqueeze(0))
            self.linear.bias.copy_(torch.tensor([b_init]))
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def calculate_logreg_lipschitz_constant(X: np.ndarray) -> float:
    """
    Calculates the smoothness (Lipschitz) constant of the gradient of the logistic regression loss.
    For logistic regression, the smoothness constant is (1/4) * largest eigenvalue of X^T X.
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
    Returns:
        float: Lipschitz constant
    """
    # Compute X^T X
    XT_X = X.T @ X
    # Compute largest eigenvalue
    eigvals = np.linalg.eigvalsh(XT_X)
    L = 1/X.shape[0]*0.25 * np.max(eigvals)
    return float(L)

def calculate_l2_regularization_lipschitz_constant() -> float:
    """
    Calculates the Lipschitz constant of the gradient of the L2 regularization term.
    For L2 regularization (||w||^2), the gradient is 2 * w,
    so the Lipschitz constant is 2.

    Args:
        lambd (float): Regularization strength (lambda)

    Returns:
        float: Lipschitz constant for the L2 regularization term
    """
    return 2.0

class LogRegScalarization(Scalarization):
    def __init__(
        self,
        num_objs: int,
        X: MatrixLike,
        y: ArrayLike,
        fair_feat: str,
        max_iter: int = 100,
        tol: float = 10**-4,
        lower_bound_estimate: str | float = 0.01, # Options: "zero", "lipshitz", or a float value
    ) -> None:
        super(LogRegScalarization, self).__init__(num_objs)
        self.fair_feat = fair_feat
        self.fair_att = sorted(X[fair_feat].unique())
        self.__M = num_objs

        self.X = X
        self.y = y

        lambd = math.exp(-100)

        self.lower_bound_estimate = lower_bound_estimate
        if self.lower_bound_estimate == "lipshitz":
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

    @property
    def objs_lower(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        # This is a placeholder, actual implementation may vary
        if self.__objs_lower is not None:
            return self.__objs_lower
        else:
            if self.lower_bound_estimate == "lipshitz":
                J = np.array(self.gradient)
                grad_w = self.w@J
                L = self.w@self.L
                objs_delta = 1/L*J@grad_w - 1/2*self.L/(L**2)*(grad_w@grad_w)
                self.__objs_lower = self.objs - objs_delta
                print("L", self.L)
                print("Objs delta:", objs_delta)
                print("Grad squar:", np.array([grad@grad for l, grad in zip(self.L, self.gradient)]))
            elif self.lower_bound_estimate == "zero":
                self.__objs_lower = np.zeros(self.M)
            else:
                self.__objs_lower = self.objs - self.lower_bound_estimate * abs(self.objs)
            return self.__objs_lower

    def get_gradient(
        self,
        X_train: MatrixLike,
        y_train: ArrayLike,
        model: Any
    ) -> npt.NDArray[np.float64]:
        X_tensor = torch.tensor(np.asarray(X_train), dtype=torch.float32)
        y_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
        
        #w = torch.tensor(model.coef_, dtype=torch.float32, requires_grad=True)
        #b = torch.tensor(model.intercept_, dtype=torch.float32)

        #logits = X_tensor @ w.T + b  # shape: (n_samples, 1)
        #preds = torch.sigmoid(logits)

        #loss = nn.functional.binary_cross_entropy(preds, y_tensor)

        w = model.coef_.flatten()       # shape: (n_features,)
        b = model.intercept_.item()

        pmodel = TorchLogReg(w, b)

        # 5. Enable gradient tracking
        for param in pmodel.parameters():
            param.requires_grad = True

        criterion = nn.BCELoss(reduction='mean')
        output = pmodel(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()

        grad_w = pmodel.linear.weight.grad.detach().numpy()
        grad_b = pmodel.linear.bias.grad.item()
        return np.concatenate([grad_w.flatten(), [grad_b]])

    def training(
        self,
        weight: npt.NDArray[np.float64]
    ) -> tuple[LogisticRegression, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[LogisticRegression, npt.NDArray[np.float64]]:
        
        self.__objs_lower = None  # Reset lower bound
        
        if self.M == 2:
            fair_weight = weight
        elif self.M == 3:
            fair_weight = (weight[:-1]+EPS) / (1 - weight[-1] + EPS)
            self.model.set_params(C= (1 - weight[-1] + EPS) / (weight[-1] + EPS))
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

        if self.M == 3:
            objs[-1] = squared_norm(self.model.coef_)

        if self.lower_bound_estimate == "lipshitz":
            group_values = self.X[self.fair_feat].to_numpy()
            gradients = []
            for g in self.fair_att:
                mask = (group_values == g)
                Xg = self.X.to_numpy()[mask]
                yg = self.y.to_numpy()[mask]

                grads = self.get_gradient(Xg, yg, self.model).squeeze()
                gradients.append(grads)

            if self.M == 3:
                grads = 2*np.concatenate([self.model.coef_.flatten(), [0]])
                gradients.append(grads)
            return self.model, objs, gradients
        return self.model, objs

# Assuming the rest of your provided code (MLP, MLPScalarization) is here

def get_hessian_vector_product(loss, params, v):
    """
    Computes the Hessian-vector product H*v for a given loss and parameters.
    """
    # First backpropagation: get the gradient of the loss w.r.t. parameters.
    # create_graph=True is essential to build the graph for the second derivative.
    grad_L = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten the gradients to be a single vector
    flat_grad_L = torch.cat([g.view(-1) for g in grad_L])
    
    # Second backpropagation: get the gradient of (grad_L . v) w.r.t. parameters.
    # This is equivalent to H*v.
    grad_L_v = torch.dot(flat_grad_L, v)
    hvp = torch.autograd.grad(grad_L_v, params)
    
    # Flatten the HVP result to be a single vector
    flat_hvp = torch.cat([h.view(-1) for h in hvp])
    
    return flat_hvp

def calculate_smoothness_lipschitz_constant(model, criterion, X_tensor, y_tensor, num_iterations=20):
    """
    Estimates the Lipschitz constant of smoothness (max eigenvalue of the Hessian)
    using the power iteration method.
    """
    # 1. Get the model parameters
    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    # 2. Initialize a random vector v for power iteration
    v = torch.randn(num_params, dtype=torch.float32)
    v = v / torch.norm(v) # Normalize

    # 3. Calculate the loss at the current model state
    model.eval() # Ensure model is in eval mode

    # 4. Power iteration loop
    for _ in range(num_iterations):
        y_pred = model(X_tensor).squeeze()
        loss = criterion(y_pred, y_tensor)
        # Calculate the Hessian-vector product
        Hv = get_hessian_vector_product(loss, params, v)
        
        # Update the estimate of the eigenvector
        v = Hv / torch.norm(Hv)

    # 5. Calculate the largest eigenvalue (Rayleigh quotient)
    # Re-calculate the final Hv to ensure it's up-to-date with the final v
    y_pred = model(X_tensor).squeeze()
    loss = criterion(y_pred, y_tensor)
    Hv = get_hessian_vector_product(loss, params, v)
    eigenvalue = torch.dot(v, Hv) / torch.dot(v, v)
    
    return eigenvalue.item()


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
        max_iter: int = 200,
        num_objs: int = 2,
        lower_bound_estimate: str | float = 0.01, # Options: "zero", "lipshitz", or a float value
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
        self.num_epochs = max_iter

        self.model = MLP(input_dim, hidden_dim, output_dim)

        self.lower_bound_estimate = lower_bound_estimate
        # Handle lower_bound_estimate option
        if lower_bound_estimate == "lipshitz":
            self.L = np.zeros(len(self.fair_att))
            group_values = self.X[self.fair_feat].to_numpy()
            for g in self.fair_att:
                mask = (group_values == g)
                Xg = torch.tensor(self.X.to_numpy()[mask], dtype=torch.float32)
                yg = torch.tensor(self.y.to_numpy()[mask], dtype=torch.float32)

                lip_criterion = nn.BCELoss(reduction="mean") 
                
                self.L[g] = calculate_smoothness_lipschitz_constant(
                    self.model, 
                    lip_criterion, 
                    Xg, 
                    yg
                )
            print("Lipschitz constants for each group:", self.L)
        else:
            self.L = None

    @property
    def objs_lower(self) -> npt.NDArray[np.float64]:
        """Returns a lower estimative of the objective values.

        Returns:
            np.ndarray: Lower estimative for each objective function.
        """
        # This is a placeholder, actual implementation may vary
        if self.__objs_lower is not None:
            return self.__objs_lower
        else:
            if self.lower_bound_estimate == "lipshitz":
                w_gradient = self.w@np.array([self.gradient[idx] for idx in range(self.M)])
                objs_delta = 1/(2*self.w@self.L)*w_gradient@w_gradient
                self.__objs_lower = self.objs - objs_delta
            elif self.lower_bound_estimate == "zero":
                self.__objs_lower = np.zeros(self.M)
            else:
                self.__objs_lower = self.objs - self.lower_bound_estimate * abs(self.objs)
            return self.__objs_lower

    def training(
        self,
        weight:npt.NDArray[np.float64]
    ) -> tuple[MLP, npt.NDArray[np.float64], npt.NDArray[np.float64]] | tuple[MLP, npt.NDArray[np.float64]]:
        self.__objs_lower = None  # Reset lower bound
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
            if self.lower_bound_estimate == "lipshitz":
                mask = self.X[self.fair_feat] == feat
                X_group = self.X_tensor[mask.to_numpy()]
                y_group = self.y_tensor[mask.to_numpy()]

                self.model.zero_grad()

                # Enable gradient computation
                X_group.requires_grad = True

                y_pred_group = self.model(X_group).squeeze()
                loss_group = nn.BCELoss(reduction="mean")(y_pred_group, y_group)
                loss_group.backward()

                # Get gradients of the model parameters
                grad_vec = np.concatenate([
                    param.grad.view(-1).detach().numpy() for param in self.model.parameters()
                ])
                gradients.append(grad_vec)

            fair_weight = np.zeros(self.M)
            fair_weight[i] = 1
            sample_weight = self.X[self.fair_feat].replace(
                {ff: fw for ff, fw in zip(self.fair_att, fair_weight)}
            )
            objs[i] = log_loss(
                self.y, y_pred.detach().numpy(), sample_weight=sample_weight
            )

        if self.lower_bound_estimate == "lipshitz":
            # Stack gradients across all objectives
            gradient = np.vstack(gradients)
            return self.model, objs, gradient
        return self.model, objs
