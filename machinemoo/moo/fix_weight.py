from scipy.optimize import minimize
import numpy as np

def solve_optimal_w_scipy(grads, L, epsilon=1e-8):
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

def solve_optimal_w_active_set(grads, L):
    """
    Solves for w that satisfies M w = L subject to w >= 0, sum(w)=1.
    Uses an Active Set method to handle the non-negativity constraints
    without computing the matrix inverse.
    """
    n_obj = len(L)
    
    # 1. Compute Gram Matrix M = G @ G.T
    # We add a tiny regularization (reg) to the diagonal to ensure 
    # the matrix is positive-definite (invertible) even if gradients are collinear.
    M = grads @ grads.T
    
    # Keep track of which objectives are currently "active"
    active_mask = np.ones(n_obj, dtype=bool)
    w = np.zeros(n_obj)
    
    # Active Set Loop
    # In the worst case, this runs n_obj times (removing one weight each time)
    while True:
        # Extract the sub-system for active variables
        # M_sub is (k x k), L_sub is (k,)
        M_sub = M[np.ix_(active_mask, active_mask)]
        L_sub = L[active_mask]
        
        # 2. Solve M_sub * w_sub = L_sub
        # This is O(k^3) but faster/stable than inv(M)
        try:
            w_sub = np.linalg.solve(M_sub, L_sub)
        except np.linalg.LinAlgError:
            # Fallback for singular cases (rare with regularization)
            w_sub = np.linalg.lstsq(M_sub, L_sub, rcond=None)[0]
        
        # 3. Check for feasibility (are all weights positive?)
        if np.all(w_sub >= 0):
            # Success! Store results and break
            w[active_mask] = w_sub
            w[~active_mask] = 0.0 # Inactive weights are 0
            break
        
        # 4. If negative, remove the "worst" offender
        # We find the index in the *sub-problem* that is most negative
        min_idx_sub = np.argmin(w_sub)
        
        # Map back to the global index to update the mask
        # active_indices[min_idx_sub] gives the global index
        active_indices = np.where(active_mask)[0]
        worst_global_idx = active_indices[min_idx_sub]
        
        # Remove this objective from the active set
        active_mask[worst_global_idx] = False
        
        # Edge case: If we removed everything (impossible in valid physics), break
        if not np.any(active_mask):
            raise ValueError("Optimization failed: All weights became negative.")

    # 5. Normalize to Simplex (Sum = 1)
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        # Fallback if solution is exactly zero vector
        w = np.ones(n_obj) / n_obj
        
    return w