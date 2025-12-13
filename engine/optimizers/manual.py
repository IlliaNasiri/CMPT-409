import torch
from .base import OptimizerState, EPS, GRAD_TOL, CLAMP_MIN, CLAMP_MAX


# --- Manual Optimized Factories ---

def ManualAdam(lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8):
    return ManualTwolayerAdam(lr=lr, betas=betas, eps=eps)

def ManualAdaGrad(lr: float = 1e-2, eps=1e-8):
    return ManualTwolayerAdaGrad(lr=lr, eps=eps)

def ManualSAM_Adam(lr: float = 1e-3, rho=0.05, betas=(0.9, 0.999), eps=1e-8):
    return ManualTwolayerSAM_Adam(lr=lr, rho=rho, betas=betas, eps=eps)

def ManualSAM_AdaGrad(lr: float = 1e-2, rho=0.05, eps=1e-8):
    return ManualTwolayerSAM_AdaGrad(lr=lr, rho=rho, eps=eps)

def ManualGD(lr: float = 1e-1):
    return ManualTwolayerGD(lr=lr)

def ManualNGD(lr: float = 1e-1):
    return ManualTwolayerNGD(lr=lr)

def ManualSAM(lr: float = 1e-1, rho=0.05):
    return ManualTwolayerSAM(lr=lr, rho=rho)

def ManualSAM_NGD(lr: float = 1e-1, rho=0.05):
    return ManualTwolayerSAM_NGD(lr=lr, rho=rho)

# -----------------------------------------------------------------------------
# Shared Logic
# -----------------------------------------------------------------------------

def _compute_grads(W1: torch.Tensor, W2: torch.Tensor, X: torch.Tensor, y: torch.Tensor, clamp_min: float = CLAMP_MIN, clamp_max: float = CLAMP_MAX):
    """
    Computes gradients manually for the linear TwoLayerModel: f(x) = W2 @ (W1 @ x)
    Returns (W1_grad, W2_grad)
    """
    # 1. Forward
    # Z = X @ W1.T  (N, k)
    # S = Z @ W2.T  (N, 1)
    Z = torch.matmul(X, W1.T)
    S = torch.matmul(Z, W2.T)

    # 2. Loss Gradient (Exponential Loss)
    # dL/dS = -1/N * y * exp(-clamp(y*S))
    y = y.view(-1, 1)
    margins = torch.clamp(y * S, clamp_min, clamp_max)
    coeffs = -torch.exp(-margins) * y
    d_scores = coeffs / X.shape[0]

    # 3. Backward
    # dL/dW2 = d_scores.T @ Z   -> (1, N) @ (N, k) -> (1, k)
    W2_grad = torch.matmul(d_scores.T, Z)

    # dL/dZ  = d_scores @ W2    -> (N, 1) @ (1, k) -> (N, k)
    d_Z = torch.matmul(d_scores, W2)

    # dL/dW1 = d_Z.T @ X        -> (k, N) @ (N, D) -> (k, D)
    W1_grad = torch.matmul(d_Z.T, X)

    return W1_grad, W2_grad


# -----------------------------------------------------------------------------
# Manual Optimizer Implementations
# -----------------------------------------------------------------------------

class ManualTwolayerAdam(OptimizerState):
    """Fused Adam for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.default_lr = lr
        self.betas = betas
        self.eps = eps
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        if self.state is None:
            W1, W2 = model.net[0].weight, model.net[1].weight
            self.state = {
                'step_t': 0,
                'm1': torch.zeros_like(W1), 'v1': torch.zeros_like(W1),
                'm2': torch.zeros_like(W2), 'v2': torch.zeros_like(W2),
            }

        s = self.state
        s['step_t'] += 1
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            W1_grad, W2_grad = _compute_grads(W1, W2, X, y)

            # Adam Update
            beta1, beta2 = self.betas
            bias_corr1 = 1 - beta1 ** s['step_t']
            bias_corr2 = 1 - beta2 ** s['step_t']
            step_size = lr / bias_corr1

            # Use scalar exponentiation instead of math.sqrt
            # This works for both Python floats and PyTorch tensors
            bias_corr2_sqrt = bias_corr2 ** 0.5

            # W1
            s['m1'].mul_(beta1).add_(W1_grad, alpha=1 - beta1)
            s['v1'].mul_(beta2).addcmul_(W1_grad, W1_grad, value=1 - beta2)
            denom1 = (s['v1'].sqrt() / bias_corr2_sqrt).add_(self.eps)
            W1.addcdiv_(s['m1'], denom1, value=-step_size)

            # W2
            s['m2'].mul_(beta1).add_(W2_grad, alpha=1 - beta1)
            s['v2'].mul_(beta2).addcmul_(W2_grad, W2_grad, value=1 - beta2)
            denom2 = (s['v2'].sqrt() / bias_corr2_sqrt).add_(self.eps)
            W2.addcdiv_(s['m2'], denom2, value=-step_size)


class ManualTwolayerAdaGrad(OptimizerState):
    """Fused Adagrad for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-2, eps=1e-8):
        self.default_lr = lr
        self.eps = eps
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        if self.state is None:
            W1, W2 = model.net[0].weight, model.net[1].weight
            self.state = {
                # Sum of squared gradients accumulator
                'sum_sq1': torch.zeros_like(W1),
                'sum_sq2': torch.zeros_like(W2),
            }

        s = self.state
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            W1_grad, W2_grad = _compute_grads(W1, W2, X, y)

            # Adagrad Update
            # W1
            s['sum_sq1'].addcmul_(W1_grad, W1_grad)
            std1 = s['sum_sq1'].sqrt().add_(self.eps)
            W1.addcdiv_(W1_grad, std1, value=-lr)

            # W2
            s['sum_sq2'].addcmul_(W2_grad, W2_grad)
            std2 = s['sum_sq2'].sqrt().add_(self.eps)
            W2.addcdiv_(W2_grad, std2, value=-lr)


class ManualTwolayerSAM_Adam(OptimizerState):
    """Fused SAM-Adam for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-3, rho=0.05, betas=(0.9, 0.999), eps=1e-8):
        self.default_lr = lr
        self.rho = rho
        self.betas = betas
        self.eps = eps
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        if self.state is None:
            W1, W2 = model.net[0].weight, model.net[1].weight
            self.state = {
                'step_t': 0,
                'm1': torch.zeros_like(W1), 'v1': torch.zeros_like(W1),
                'm2': torch.zeros_like(W2), 'v2': torch.zeros_like(W2),
            }

        s = self.state
        s['step_t'] += 1
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. First Step: Compute Gradients at current W
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. Compute SAM Perturbation
            # Global norm over all params
            gnorm = torch.sqrt(g1.norm()**2 + g2.norm()**2)
            if gnorm > GRAD_TOL:
                scale = self.rho / gnorm

                # Perturbed weights (temporary)
                W1_adv = W1 + g1 * scale
                W2_adv = W2 + g2 * scale

                # 3. Second Step: Compute Gradients at perturbed W_adv
                g1_adv, g2_adv = _compute_grads(W1_adv, W2_adv, X, y)

                # 4. Adam Update on ORIGINAL weights using ADV gradients
                beta1, beta2 = self.betas
                bias_corr1 = 1 - beta1 ** s['step_t']
                bias_corr2 = 1 - beta2 ** s['step_t']
                step_size = lr / bias_corr1

                bias_corr2_sqrt = bias_corr2 ** 0.5

                # W1
                s['m1'].mul_(beta1).add_(g1_adv, alpha=1 - beta1)
                s['v1'].mul_(beta2).addcmul_(g1_adv, g1_adv, value=1 - beta2)
                denom1 = (s['v1'].sqrt() / bias_corr2_sqrt).add_(self.eps)
                W1.addcdiv_(s['m1'], denom1, value=-step_size)

                # W2
                s['m2'].mul_(beta1).add_(g2_adv, alpha=1 - beta1)
                s['v2'].mul_(beta2).addcmul_(g2_adv, g2_adv, value=1 - beta2)
                denom2 = (s['v2'].sqrt() / bias_corr2_sqrt).add_(self.eps)
                W2.addcdiv_(s['m2'], denom2, value=-step_size)


class ManualTwolayerSAM_AdaGrad(OptimizerState):
    """Fused SAM-Adagrad for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-2, rho=0.05, eps=1e-8):
        self.default_lr = lr
        self.rho = rho
        self.eps = eps
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        if self.state is None:
            W1, W2 = model.net[0].weight, model.net[1].weight
            self.state = {
                'sum_sq1': torch.zeros_like(W1),
                'sum_sq2': torch.zeros_like(W2),
            }

        s = self.state
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. First Step: Gradient at current W
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. SAM Perturbation
            gnorm = torch.sqrt(g1.norm()**2 + g2.norm()**2)
            if gnorm > GRAD_TOL:
                scale = self.rho / gnorm

                W1_adv = W1 + g1 * scale
                W2_adv = W2 + g2 * scale

                # 3. Second Step: Gradient at perturbed W_adv
                g1_adv, g2_adv = _compute_grads(W1_adv, W2_adv, X, y)

                # 4. Adagrad Update on ORIGINAL weights
                # W1
                s['sum_sq1'].addcmul_(g1_adv, g1_adv)
                std1 = s['sum_sq1'].sqrt().add_(self.eps)
                W1.addcdiv_(g1_adv, std1, value=-lr)

                # W2
                s['sum_sq2'].addcmul_(g2_adv, g2_adv)
                std2 = s['sum_sq2'].sqrt().add_(self.eps)
                W2.addcdiv_(g2_adv, std2, value=-lr)


class ManualTwolayerGD(OptimizerState):
    """Standard Gradient Descent for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-1):
        self.default_lr = lr
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. Compute Gradients
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. Update
            W1.add_(g1, alpha=-lr)
            W2.add_(g2, alpha=-lr)


class ManualTwolayerNGD(OptimizerState):
    """Normalized Gradient Descent for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-1):
        self.default_lr = lr
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. Compute Gradients
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. Compute Global Norm
            # sqrt(||g1||^2 + ||g2||^2)
            gnorm_sq = g1.norm()**2 + g2.norm()**2
            gnorm = torch.sqrt(gnorm_sq)

            # 3. NGD Update (Global)
            if gnorm > GRAD_TOL:
                # Effective step size for normalized update
                scale = -lr / gnorm
                W1.add_(g1, alpha=scale)
                W2.add_(g2, alpha=scale)
            # else: gradient is zero, no update


class ManualTwolayerSAM(OptimizerState):
    """Sharpness-Aware Minimization (SAM) for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-1, rho=0.05):
        self.default_lr = lr
        self.rho = rho
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. Compute Gradients at current W
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. SAM Perturbation (Global Norm)
            gnorm = torch.sqrt(g1.norm()**2 + g2.norm()**2)

            if gnorm > GRAD_TOL:
                scale = self.rho / gnorm

                # Perturb weights
                W1_adv = W1 + g1 * scale
                W2_adv = W2 + g2 * scale

                # 3. Compute Gradients at perturbed W_adv
                g1_adv, g2_adv = _compute_grads(W1_adv, W2_adv, X, y)

                # 4. Standard GD Update on ORIGINAL weights using ADV gradients
                W1.add_(g1_adv, alpha=-lr)
                W2.add_(g2_adv, alpha=-lr)


class ManualTwolayerSAM_NGD(OptimizerState):
    """SAM + Normalized GD for Linear TwoLayerModel."""
    def __init__(self, lr: float = 1e-1, rho=0.05):
        self.default_lr = lr
        self.rho = rho
        self.state = None

    def reset(self):
        self.state = None

    def step(self, model, X, y, lr: float):
        W1, W2 = model.net[0].weight, model.net[1].weight

        with torch.no_grad():
            # 1. Compute Gradients at current W
            g1, g2 = _compute_grads(W1, W2, X, y)

            # 2. SAM Perturbation (Global Norm)
            gnorm = torch.sqrt(g1.norm()**2 + g2.norm()**2)

            if gnorm > GRAD_TOL:
                scale = self.rho / gnorm

                # Perturb weights
                W1_adv = W1 + g1 * scale
                W2_adv = W2 + g2 * scale

                # 3. Compute Gradients at perturbed W_adv
                g1_adv, g2_adv = _compute_grads(W1_adv, W2_adv, X, y)

                # 4. NGD Update (Global Norm of ADV gradients)
                gnorm_adv = torch.sqrt(g1_adv.norm()**2 + g2_adv.norm()**2)

                if gnorm_adv > GRAD_TOL:
                    update_scale = -lr / gnorm_adv
                    W1.add_(g1_adv, alpha=update_scale)
                    W2.add_(g2_adv, alpha=update_scale)
