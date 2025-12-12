import torch

# -----------------------------------------------------------------------------
# First-Order Optimizers (GD, NGD, SAM, SAM+NGD)
# -----------------------------------------------------------------------------
# All optimizers accept a Model and loss_fn, and perform backpropagation
# Everything is PyTorch - no NumPy support

def step_gd(model, X, y, lr, loss_fn):
    """
    Gradient Descent step with backpropagation.

    Args:
        model: Model instance (PyTorch)
        X: Input data (torch.Tensor)
        y: Labels (torch.Tensor)
        lr: Learning rate
        loss_fn: Reusable loss function (ExponentialLoss instance)
    """
    # Zero gradients
    model.zero_grad()

    # Forward pass
    scores = model.forward(X)

    # Compute loss and gradients
    loss = loss_fn(scores, y)
    loss.backward()

    # Update parameters: w = w - lr * grad
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad


def step_ngd_stable(model, X, y, lr, loss_fn):
    """
    Normalized Gradient Descent with numerical stability.

    Note: Despite the name, this implements normalized GD (softmax weighting),
    not true Natural Gradient Descent.

    Args:
        model: Model instance (PyTorch)
        X: Input data (torch.Tensor)
        y: Labels (torch.Tensor)
        lr: Learning rate
        loss_fn: Reusable loss function (ExponentialLoss instance)
    """
    # Zero gradients
    model.zero_grad()

    # Forward pass
    scores = model.forward(X)

    # Compute softmax-weighted loss
    margins = y * scores
    neg_margins = -margins

    # Numerical stability: subtract max before exp
    shift = torch.max(neg_margins)
    exps = torch.exp(neg_margins - shift)
    softmax_weights = exps / torch.sum(exps)

    # Weighted exponential loss
    loss = torch.sum(softmax_weights * torch.exp(-margins))
    loss.backward()

    # Update parameters
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad


def step_sam_stable(model, X, y, lr, loss_fn, rho=0.05):
    """
    Sharpness-Aware Minimization (SAM) with backpropagation.

    Performs two forward/backward passes:
    1. Compute gradient at current point
    2. Perturb in direction of gradient
    3. Compute gradient at perturbed point
    4. Update using gradient from perturbed point

    Args:
        model: Model instance (PyTorch)
        X: Input data (torch.Tensor)
        y: Labels (torch.Tensor)
        lr: Learning rate
        loss_fn: Reusable loss function (ExponentialLoss instance)
        rho: SAM perturbation radius (default 0.05)
    """
    # First forward/backward: compute gradient at current point
    model.zero_grad()
    scores = model.forward(X)
    loss = loss_fn(scores, y)
    loss.backward()

    # Compute perturbation: eps = rho * grad / ||grad||
    # Save current parameters and perturb
    original_params = []
    with torch.no_grad():
        for param in model.parameters():
            original_params.append(param.clone())
            if param.grad is not None:
                grad_norm = param.grad.norm() + 1e-12
                param.add_(param.grad, alpha=rho / grad_norm)

    # Second forward/backward: compute gradient at perturbed point
    model.zero_grad()
    scores_adv = model.forward(X)
    loss_adv = loss_fn(scores_adv, y)
    loss_adv.backward()

    # Restore original parameters and apply update using adversarial gradient
    with torch.no_grad():
        for param, param_orig in zip(model.parameters(), original_params):
            grad_adv = param.grad
            param.copy_(param_orig)
            if grad_adv is not None:
                param -= lr * grad_adv


def step_sam_ngd_stable(model, X, y, lr, loss_fn, rho=0.05):
    """
    SAM + Normalized GD: First perturb with SAM, then use normalized GD for the update.

    Args:
        model: Model instance (PyTorch)
        X: Input data (torch.Tensor)
        y: Labels (torch.Tensor)
        lr: Learning rate
        loss_fn: Reusable loss function (ExponentialLoss instance)
        rho: SAM perturbation radius (default 0.05)
    """
    # SAM perturbation step
    model.zero_grad()
    scores = model.forward(X)
    loss = loss_fn(scores, y)
    loss.backward()

    # Save original parameters and perturb
    original_params = []
    with torch.no_grad():
        for param in model.parameters():
            original_params.append(param.clone())
            if param.grad is not None:
                grad_norm = param.grad.norm() + 1e-12
                param.add_(param.grad, alpha=rho / grad_norm)

    # Normalized GD step at perturbed point
    model.zero_grad()
    scores_adv = model.forward(X)
    margins = y * scores_adv
    neg_margins = -margins

    # Softmax weighting
    shift = torch.max(neg_margins)
    exps = torch.exp(neg_margins - shift)
    softmax_weights = exps / torch.sum(exps)

    # Weighted loss
    loss_weighted = torch.sum(softmax_weights * torch.exp(-margins))
    loss_weighted.backward()

    # Restore original parameters and apply normalized GD update
    with torch.no_grad():
        for param, param_orig in zip(model.parameters(), original_params):
            grad_adv = param.grad
            param.copy_(param_orig)
            if grad_adv is not None:
                param -= lr * grad_adv
