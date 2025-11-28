from scipy.optimize import minimize
import numpy as np

def solve_optimal_w(grads, L, epsilon=1e-8):
    """
    Finds w that minimizes (w_grads @ w_grads) / (w @ L)
    subject to w being a simplex.
    """
    n_obj = len(L)
    
    # Precompute Gram Matrix M = G @ G.T (Shape: n_obj x n_obj)
    # This represents the dot products between all pairs of objective gradients.
    M = grads @ grads.T

    def objective(w):
        # Numerator: w^T M w
        # This is equivalent to || w @ grads ||^2
        numerator = w @ M @ w
        
        # Denominator: w @ L
        denominator = w @ L
        
        return numerator / (denominator + epsilon)

    # Jacobian of the objective (optional, speeds up convergence)
    def jacobian(w):
        num = w @ M @ w
        den = w @ L + epsilon
        
        # d/dw (N/D) = (D * N' - N * D') / D^2
        # N' = 2Mw (gradient of quadratic form)
        # D' = L   (gradient of linear form)
        grad_num = 2 * (M @ w)
        grad_den = L
        
        return (den * grad_num - num * grad_den) / (den**2)

    # Constraints: sum(w) = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Bounds: 0 <= w <= 1
    bounds = tuple((0.0, 1.0) for _ in range(n_obj))
    
    # Initial guess: Uniform distribution
    w0 = np.ones(n_obj) / n_obj
    
    result = minimize(
        objective, 
        w0, 
        method='SLSQP', 
        jac=jacobian,
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )
    
    return result.x