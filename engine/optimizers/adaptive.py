import torch
from .base import make_adaptive_optimizer

# -----------------------------------------------------------------------------
# Public Optimizer Constructors
# -----------------------------------------------------------------------------

def Adam(lr: float, device: str = 'cpu', betas=(0.9, 0.999), eps=1e-8):
    """Returns a StatefulOptimizer using Adam."""
    # Closure to bind 'lr' and ignore the step_lr passed by StatefulOptimizer
    return make_adaptive_optimizer(
        torch.optim.Adam, lr=lr, device=device, betas=betas, eps=eps
    )

def Adagrad(lr: float, device: str = 'cpu', eps=1e-8):
    """Returns a StatefulOptimizer using Adagrad."""
    return make_adaptive_optimizer(
        lambda D, _, **kwargs: make_adagrad_step(D, lr, **kwargs),
        device=device, eps=eps
    )

def SAM_Adam(lr: float, device: str = 'cpu', rho=0.05, betas=(0.9, 0.999), eps=1e-8):
    """Returns a StatefulOptimizer using SAM-Adam."""
    return make_adaptive_optimizer(
        lambda D, _, **kwargs: make_sam_adam_step(D, lr, **kwargs),
        device=device, rho=rho, betas=betas, eps=eps
    )

def SAM_Adagrad(lr: float, device: str = 'cpu', rho=0.05, eps=1e-8):
    """Returns a StatefulOptimizer using SAM-Adagrad."""
    return make_adaptive_optimizer(
        lambda D, _, **kwargs: make_sam_adagrad_step(D, lr, **kwargs),
        device=device, rho=rho, eps=eps
    )
# -----------------------------------------------------------------------------
# Internal Factory Functions
# -----------------------------------------------------------------------------

def make_adam_step(D: int, lr: float, device='cpu', betas=(0.9, 0.999), eps=1e-8):
    """
    Internal factory for Adam step function.
    """
    # Persistent state strictly typed to float64 and specific device
    print(f"{D}, dtype=torch.float64, device={device}")
    w_torch = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([w_torch], lr=lr, betas=betas, eps=eps)

    def step(w, X, y, lr_unused=None):
        nonlocal w_torch, opt

        # Load external weights into internal state
        # Assumes w, X, y are torch.Tensors on the correct device
        with torch.no_grad():
            w_torch.copy_(w)
        w_torch.requires_grad_(True)

        # Standard loss computation
        margins = y * (X @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        return w_torch.detach()

    return step


def make_adagrad_step(D: int, lr: float, device='cpu', eps=1e-8):
    """
    Internal factory for Adagrad step function.
    """
    w_torch = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adagrad([w_torch], lr=lr, eps=eps)

    def step(w, X, y, lr_unused=None):
        nonlocal w_torch, opt

        with torch.no_grad():
            w_torch.copy_(w)
        w_torch.requires_grad_(True)

        margins = y * (X @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        return w_torch.detach()

    return step


def make_sam_adam_step(D: int, lr: float, device='cpu', rho=0.05, betas=(0.9, 0.999), eps=1e-8):
    """
    Internal factory for SAM-Adam step function.
    """
    w_torch = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adam([w_torch], lr=lr, betas=betas, eps=eps)

    def step(w, X, y, lr_unused=None):
        nonlocal w_torch, opt

        # 1. Sync internal state with current weights
        with torch.no_grad():
            w_torch.copy_(w)
        w_torch.requires_grad_(True)

        # 2. Compute gradients at w for perturbation
        margins = y * (X @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))
        
        # We use autograd.grad to get gradients without populating .grad field yet
        grad = torch.autograd.grad(loss, w_torch)[0]

        # 3. Compute perturbation (w_adv)
        g_norm = grad.norm() + 1e-12
        w_adv = (w_torch + rho * grad / g_norm).detach()

        # 4. Move internal state to w_adv to compute update gradients
        with torch.no_grad():
            w_torch.copy_(w_adv)
        w_torch.requires_grad_(True)

        margins_adv = y * (X @ w_torch)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))

        opt.zero_grad()
        loss_adv.backward()  # Populates w_torch.grad with gradient at w_adv
        
        # 5. Restore w_torch to original w BEFORE stepping
        # We want to update the original point using the gradient from the adversarial point
        with torch.no_grad():
            w_torch.copy_(w)

        opt.step()

        return w_torch.detach()

    return step


def make_sam_adagrad_step(D: int, lr: float, device='cpu', rho=0.05, eps=1e-8):
    """
    Internal factory for SAM-Adagrad step function.
    """
    w_torch = torch.zeros(D, dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.Adagrad([w_torch], lr=lr, eps=eps)

    def step(w, X, y, lr_unused=None):
        nonlocal w_torch, opt

        with torch.no_grad():
            w_torch.copy_(w)
        w_torch.requires_grad_(True)

        # Gradient at w
        margins = y * (X @ w_torch)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))
        grad = torch.autograd.grad(loss, w_torch)[0]

        # Perturbation
        g_norm = grad.norm() + 1e-12
        w_adv = (w_torch + rho * grad / g_norm).detach()

        # Gradient at w_adv
        with torch.no_grad():
            w_torch.copy_(w_adv)
        w_torch.requires_grad_(True)

        margins_adv = y * (X @ w_torch)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))

        opt.zero_grad()
        loss_adv.backward()

        # Update original w
        with torch.no_grad():
            w_torch.copy_(w)
            
        opt.step()

        return w_torch.detach()

    return step


