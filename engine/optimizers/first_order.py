import torch

# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support

# Machine epsilon for float64 (Double Precision)
EPS = 1e-16 

def _linear_grad_exponential(X, y, w, clamp_min=-50, clamp_max=100):
    """
    Helper to compute the gradient of Exponential Loss for a linear model manually.
    """
    scores = X @ w
    
    # Ensure y matches scores shape (N,) or (N,1)
    y = y.view_as(scores)
    
    margins = y * scores
    margins_clamped = torch.clamp(margins, clamp_min, clamp_max)
    exp_neg_margins = torch.exp(-margins_clamped)
    
    # CRITICAL FIX: The scalar term must be (N, 1) to broadcast against X (N, D)
    # regardless of whether scores was (N,) or (N, 1).
    scalar_term = (y * exp_neg_margins).view(-1, 1)
    
    # Gradient: -mean(scalar_term * X)
    grad = -torch.mean(scalar_term * X, dim=0)
    
    # Ensure gradient shape matches weight shape
    return grad.view_as(w)


def step_gd(model, X, y, lr, loss_fn):
    """Gradient Descent step."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w)
            model.w -= lr * grad
    else:
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad


def step_ngd_stable(model, X, y, lr, loss_fn):
    """Normalized Gradient Descent."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            scores = X @ model.w
            y = y.view_as(scores)
            
            margins = y * scores
            neg_margins = -margins

            # Softmax calculation
            shift = torch.max(neg_margins)
            exps = torch.exp(neg_margins - shift)
            S = torch.sum(exps)
            
            if S > EPS:
                sum_exps_sq = torch.sum(exps * exps)
                term_bracket = (sum_exps_sq / S) - 2 * exps
                scaling = torch.exp(-shift) / S
                
                coeffs = y * (exps * scaling * term_bracket)
                
                # Robust matrix multiply:
                # Treat coeffs as (N, 1) and use Transpose to get (1, N) @ (N, D) -> (1, D)
                # This works for both 1D and 2D weight vectors.
                coeffs = coeffs.view(-1, 1)
                grad = torch.matmul(coeffs.T, X)
                
                model.w -= lr * grad.view_as(model.w)
    else:
        model.zero_grad()
        scores = model.forward(X)
        y_view = y.view_as(scores)
        margins = y_view * scores
        neg_margins = -margins

        shift = torch.max(neg_margins)
        exps = torch.exp(neg_margins - shift)
        softmax_weights = exps / torch.sum(exps)

        loss = torch.sum(softmax_weights * torch.exp(-margins))
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad


def step_sam_stable(model, X, y, lr, loss_fn, rho=0.05):
    """Sharpness-Aware Minimization (SAM)."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w)
            
            grad_norm = grad.norm() + EPS
            eps = (rho / grad_norm) * grad
            
            w_adv = model.w + eps
            grad_adv = _linear_grad_exponential(X, y, w_adv)
            
            model.w -= lr * grad_adv
    else:
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        original_params = []
        with torch.no_grad():
            for param in model.parameters():
                original_params.append(param.clone())
                if param.grad is not None:
                    grad_norm = param.grad.norm() + EPS
                    param.add_(param.grad, alpha=rho / grad_norm)

        model.zero_grad()
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        with torch.no_grad():
            for param, param_orig in zip(model.parameters(), original_params):
                grad_adv = param.grad
                param.copy_(param_orig)
                if grad_adv is not None:
                    param -= lr * grad_adv


def step_sam_ngd_stable(model, X, y, lr, loss_fn, rho=0.05):
    """SAM + Normalized GD."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w)
            
            grad_norm = grad.norm() + EPS
            eps = (rho / grad_norm) * grad
            w_adv = model.w + eps
            
            scores = X @ w_adv
            y = y.view_as(scores)
            
            margins = y * scores
            neg_margins = -margins

            shift = torch.max(neg_margins)
            exps = torch.exp(neg_margins - shift)
            S = torch.sum(exps)

            if S > EPS:
                sum_exps_sq = torch.sum(exps * exps)
                term_bracket = (sum_exps_sq / S) - 2 * exps
                scaling = torch.exp(-shift) / S
                
                coeffs = y * (exps * scaling * term_bracket)
                coeffs = coeffs.view(-1, 1)
                
                grad_adv = torch.matmul(coeffs.T, X)

                model.w -= lr * grad_adv.view_as(model.w)
    else:
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        original_params = []
        with torch.no_grad():
            for param in model.parameters():
                original_params.append(param.clone())
                if param.grad is not None:
                    grad_norm = param.grad.norm() + EPS
                    param.add_(param.grad, alpha=rho / grad_norm)

        model.zero_grad()
        scores_adv = model.forward(X)
        
        y_view = y.view_as(scores_adv)
        margins = y_view * scores_adv
        neg_margins = -margins

        shift = torch.max(neg_margins)
        exps = torch.exp(neg_margins - shift)
        softmax_weights = exps / torch.sum(exps)

        loss_weighted = torch.sum(softmax_weights * torch.exp(-margins))
        loss_weighted.backward()

        with torch.no_grad():
            for param, param_orig in zip(model.parameters(), original_params):
                grad_adv = param.grad
                param.copy_(param_orig)
                if grad_adv is not None:
                    param -= lr * grad_adv
