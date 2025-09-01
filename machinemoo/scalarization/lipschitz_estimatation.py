
import torch
import torch.nn as nn
import numpy as np

from collections.abc import Callable, Iterable

## Calculate Lipschitz constant for Logistic Regression model from scikit-learn
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

## Calculate Lipschitz constant for NLP model using PyTorch
def get_hessian_vector_product(
        loss: torch.Tensor,
        params: Iterable[torch.Tensor],
        v: torch.Tensor
) -> torch.Tensor:
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

def calculate_torch_lipschitz_constant(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    device: torch.device | str = "cpu",
    num_iterations: int = 20,
    layer: nn.Module | None = None
) -> float:
    """
    Estimates the Lipschitz constant of smoothness (max eigenvalue of the Hessian)
    using the power iteration method.
    
    Args:
        model (nn.Module): PyTorch model.
        loss_fn (Callable): Function that computes the loss when called (closure).
        device (torch.device): Device where tensors are allocated.
        num_iterations (int, optional): Number of power iterations. Defaults to 20.
        layer (nn.Module, optional): Restrict Lipschitz estimation to a specific layer.
        
    Returns:
        float: Estimated Lipschitz constant (largest eigenvalue of Hessian).
    """
    if isinstance(device, str):
        device = torch.device(device)

    # 1. Get the model parameters
    params = list(model.parameters()) if layer is None else list(layer.parameters())
    num_params = sum(p.numel() for p in params)

    assert all(p.device == device for p in params), \
        "Parameters and loss must be on the same device"

    # 2. Initialize a random vector v for power iteration
    v = torch.randn(num_params, dtype=torch.float32, device=device)
    v = v / torch.norm(v)

    # 3. Calculate the loss at the current model state
    model.eval() # Ensure model is in eval mode

    # 4. Power iteration loop
    for _ in range(num_iterations):
        loss = loss_fn()

        # Calculate the Hessian-vector product
        Hv = get_hessian_vector_product(loss, params, v)

        # Update the estimate of the eigenvector
        v = Hv / torch.norm(Hv)

    # 5. Calculate the largest eigenvalue (Rayleigh quotient)
    # Re-calculate the final Hv to ensure it's up-to-date with the final v
    loss = loss_fn()
    Hv = get_hessian_vector_product(loss, params, v)
    eigenvalue = torch.dot(v, Hv) / torch.dot(v, v)

    return eigenvalue.item()