import torch
from .base import EPS, GRAD_TOL, CLAMP_MIN, CLAMP_MAX


# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support


def _linear_grad_exponential(X, y, w, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX):
    """
    Compute gradient of Exponential Loss optimized for GPU.
    Uses Matrix-Vector multiplication (addmm) to avoid large intermediate broadcasts.
    """
    # 1. Compute Scores: (N, D) @ (D,) -> (N,)
    scores = X @ w
    y = y.view_as(scores)
    
    # 2. Compute Scalar Term (the "weight" for each sample)
    margins = y * scores
    # In-place clamp is slightly faster if margins is not reused, but safety first
    margins_clamped = torch.clamp(margins, clamp_min, clamp_max)
    
    # exp(-m)
    exp_neg_margins = torch.exp(-margins_clamped)
    
    # scalar_term: y * exp(-m)
    # shape (N, 1)
    scalar_term = (y * exp_neg_margins).view(-1, 1)
    
    # 3. Compute Gradient via Matrix Multiplication
    # Original: -mean(scalar_term * X, dim=0) -> Broadcsts to (N, D) (Memory Heavy)
    # Optimized: -1/N * (X.T @ scalar_term)   -> MatMul (Compute Dense, Tensor Cores)
    # X.T is (D, N), scalar_term is (N, 1) -> Result (D, 1)
    
    N = X.shape[0]
    grad = torch.matmul(X.T, scalar_term)
    grad.div_(-N) # In-place division
    
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


def step_ngd_stable(model, X, y, lr, loss_fn, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """
    Normalized Gradient Descent with numerical stability.

    Uses clamped gradient computation to prevent overflow, and a hard threshold
    check to preserve NGD's constant step-size property (avoiding EPS degradation).
    """
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            if grad_norm > grad_tol:
                model.w -= lr * (grad / grad_norm)
            # else: gradient effectively zero, no update
    else:
        # General case: autograd path
        model.zero_grad()
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if grad_norm > grad_tol:
                        param -= lr * (param.grad / grad_norm)


def step_sam_stable(model, X, y, lr, loss_fn, rho=0.05, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """Sharpness-Aware Minimization (SAM) - Corrected."""
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. Compute Gradient
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            # 2. Compute Perturbation (Strictly enforcing norm = rho)
            if grad_norm > grad_tol:  # Use a tiny threshold significantly smaller than EPS
                eps = (rho / grad_norm) * grad
            else:
                eps = torch.zeros_like(grad) # Or random direction if preferred

            # 3. Adversarial Step
            w_adv = model.w + eps
            grad_adv = _linear_grad_exponential(X, y, w_adv, clamp_min, clamp_max)

            # 4. Update
            model.w -= lr * grad_adv
    else:
        # 1. Compute Gradients at current w
        model.zero_grad(set_to_none=True)
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        # Gather params with gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return
            
        grads = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # --- GLOBAL NORM CALCULATION ---
            # Efficiently compute norm across all tensors
            per_tensor_norms = torch._foreach_norm(grads, 2)
            global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))

            # Save original parameters
            original_params = [p.clone() for p in params_with_grad]

            # 2. SAM Perturbation (Global)
            if global_norm > grad_tol:
                scale = rho / global_norm
                # p = p + (rho/||g||) * g
                torch._foreach_add_(params_with_grad, grads, alpha=scale)

        # 3. Compute Gradients at adversarial point
        model.zero_grad(set_to_none=True)
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()

        # Refresh gradients (grads_adv)
        grads_adv = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # 4. Restore & Update

            # First, restore original weights: w = w_orig
            for p, p_orig in zip(params_with_grad, original_params):
                p.copy_(p_orig)

            # Then apply standard GD update using adversarial gradients
            # w = w - lr * g_adv
            # Note: SAM uses standard GD update, NOT normalized update
            torch._foreach_add_(params_with_grad, grads_adv, alpha=-lr)


def step_sam_ngd_stable(model, X, y, lr, loss_fn, rho=0.05, clamp_min=CLAMP_MIN, clamp_max=CLAMP_MAX, grad_tol=GRAD_TOL):
    """
    SAM + Normalized GD with numerical stability.
    """
    if hasattr(model, 'w') and len(list(model.parameters())) == 1:
        with torch.no_grad():
            # 1. SAM perturbation
            # Use the clamped helper to prevent overflow in exp()
            grad = _linear_grad_exponential(X, y, model.w, clamp_min, clamp_max)
            grad_norm = grad.norm()

            # Check against a tiny tolerance to avoid div-by-zero
            # 1e-30 is safe for float64 and allows training to continue much longer
            if grad_norm > grad_tol:
                # Strictly enforce radius = rho
                eps = (rho / grad_norm) * grad
                w_adv = model.w + eps

                # 2. NGD update at adversarial point
                grad_adv = _linear_grad_exponential(X, y, w_adv, clamp_min, clamp_max)
                grad_adv_norm = grad_adv.norm()

                if grad_adv_norm > grad_tol:
                    # Standard NGD update
                    model.w -= lr * (grad_adv / grad_adv_norm)
            else:
                # No update, treat as zero
                pass
    else:
        # 1. Compute Gradients at current w
        model.zero_grad(set_to_none=True) # Slightly faster than zeroing tensors
        scores = model.forward(X)
        loss = loss_fn(scores, y)
        loss.backward()

        # Gather params that actually have gradients
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if not params_with_grad:
            return

        grads = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # --- GLOBAL NORM (Optimized) ---
            # torch._foreach_norm computes norms of chunks efficiently
            # We stack the results (scalars) and norm that small vector
            per_tensor_norms = torch._foreach_norm(grads, 2)
            global_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms))

            # Store original parameters (clone is unavoidable for SAM)
            original_params = [p.clone() for p in params_with_grad]

            # 2. SAM Perturbation (Fused)
            if global_norm > grad_tol:
                scale = rho / global_norm
                # Fused add: p = p + scale * g
                torch._foreach_add_(params_with_grad, grads, alpha=scale)

        # 3. Compute Gradients at adversarial point
        model.zero_grad(set_to_none=True)
        scores_adv = model.forward(X)
        loss_adv = loss_fn(scores_adv, y)
        loss_adv.backward()
        
        # Refresh grads list (in case graph changed, though unlikely)
        grads_adv = [p.grad for p in params_with_grad]

        with torch.no_grad():
            # --- GLOBAL ADV NORM (Optimized) ---
            per_tensor_norms_adv = torch._foreach_norm(grads_adv, 2)
            global_adv_norm = torch.linalg.vector_norm(torch.stack(per_tensor_norms_adv))
            
            # 4. NGD Update & Restore
            
            # Restore weights first: p.copy_(p_orig)
            # There is no direct foreach_copy, but we can do it manually or 
            # if we are clever, we can calculate the net update vector.
            # But standard restore is safer.
            for p, p_orig in zip(params_with_grad, original_params):
                p.copy_(p_orig)

            if global_adv_norm > grad_tol:
                update_scale = -lr / global_adv_norm
                # Fused update: p = p - (lr/norm) * g_adv
                torch._foreach_add_(params_with_grad, grads_adv, alpha=update_scale)

