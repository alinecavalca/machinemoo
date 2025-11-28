import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Any, Optional
from machinemoo.utils.typing import MatrixLike

class TorchLogReg(nn.Module):
    """
    A simple PyTorch Logistic Regression model used for gradient estimation.
    """
    def __init__(self, w_init: np.ndarray, b_init: float) -> None:
        super().__init__()
        # w_init shape: (n_features,)
        self.linear = nn.Linear(in_features=w_init.shape[0], out_features=1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(w_init).unsqueeze(0))
            self.linear.bias.copy_(torch.tensor([b_init]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

def get_logistic_gradient(
    X: MatrixLike,
    y: MatrixLike,
    model: Any,
    sample_weight_tensor: Optional[torch.Tensor] = None
) -> npt.NDArray[np.float64]:
    """
    Safely calculates gradients using a fresh PyTorch graph every time.
    """
    # Create leaf tensors (detached from any previous graph)
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32)

    # Initialize model with numpy weights (detached)
    w = model.coef_.flatten()
    b = model.intercept_.item()
    pmodel = TorchLogReg(w, b)

    # Enable gradients only for this specific model instance
    for param in pmodel.parameters():
        param.requires_grad = True

    # Forward
    criterion = nn.BCELoss(weight=sample_weight_tensor, reduction='mean')
    output = pmodel(X_tensor).squeeze()
    
    if output.shape != y_tensor.shape:
        output = output.view_as(y_tensor)

    loss = criterion(output, y_tensor)
    
    # Backward (Calculates grads and frees graph by default)
    pmodel.zero_grad() # Clean buffer
    loss.backward()    # retain_graph=False is default

    # Extract and Detach immediately
    # PyTorch's .grad can be None in some edge cases; assert to satisfy static checkers and catch runtime problems.
    grad_w_t = pmodel.linear.weight.grad
    assert grad_w_t is not None, "Weight gradient is None after backward()"
    grad_w = grad_w_t.detach().cpu().numpy().flatten()

    grad_b_t = pmodel.linear.bias.grad
    assert grad_b_t is not None, "Bias gradient is None after backward()"
    grad_b = grad_b_t.item()
    
    # Explicit cleanup to prevent RuntimeError on loops
    del pmodel
    del loss
    del output
    
    return np.concatenate([grad_w, [grad_b]])