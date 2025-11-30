import torch
import torch.nn as nn

# ============================================================
# Two-layer network
# f(x) = sum(W2 @ (W1 @ x))
# ============================================================

class TwoLayerNet(nn.Module):
    def __init__(self, D, k=20):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(k, D) * 0.01)
        self.W2 = nn.Parameter(torch.randn(10, k) * 0.01)

    def forward(self, X):
        Z = self.W1 @ X.T               # (k, N)
        pred = self.W2 @ Z              # (10, N)
        return pred.mean(0)             # **MEAN (correct scaling)**


# ============================================================
# Loss + error
# ============================================================

def forward_two_layer(model, X):
    Z = model.W1 @ X.T
    pred = model.W2 @ Z
    return pred.mean(0)


def exponential_loss_2layer_torch(model, X, y):
    f = forward_two_layer(model, X)
    return torch.exp(-torch.clamp(y * f, min=-50, max=50)).mean()


def error_rate_2layer_torch(model, X, y):
    preds = torch.sign(forward_two_layer(model, X))
    return (preds != y).float().mean()


# ============================================================
# Helpers
# ============================================================

def flatten_grad(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ============================================================
# 1. Gradient Descent
# ============================================================

def step_gd(model, X, y, lr):
    loss = exponential_loss_2layer_torch(model, X, y)
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad

    zero_grad(model)
    return model


# ============================================================
# 2. NGD (Diagonal Fisher)
# ============================================================

def step_ngd(model, X, y, lr, eps=1e-6):
    loss = exponential_loss_2layer_torch(model, X, y)
    loss.backward()

    g = flatten_grad(model)
    F_diag = torch.abs(g) + eps           # FIXED (not g²!)
    nat_grad = g / F_diag

    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p -= lr * nat_grad[idx:idx+n].view_as(p)
            idx += n

    zero_grad(model)
    return model


# ============================================================
# 3. SAM
# ============================================================

def step_sam(model, X, y, lr, rho=0.05):

    # 1. Compute base gradient
    loss = exponential_loss_2layer_torch(model, X, y)
    loss.backward()
    g = flatten_grad(model)
    g_norm = g.norm() + 1e-12

    # Save weights
    base_weights = [p.detach().clone() for p in model.parameters()]

    # 2. Perturb
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.add_(rho * g[idx:idx+n].view_as(p) / g_norm)
            idx += n

    # 3. Gradient at perturbed weights
    zero_grad(model)
    loss2 = exponential_loss_2layer_torch(model, X, y)
    loss2.backward()

    # 4. Update original weights using perturbed gradient
    with torch.no_grad():
        idx = 0
        for p, base in zip(model.parameters(), base_weights):
            n = p.numel()

            # restore base
            p.copy_(base)

            # gradient step
            p -= lr * p.grad

            idx += n

    zero_grad(model)
    return model


# ============================================================
# 4. SAM + NGD
# ============================================================

def step_sam_ngd(model, X, y, lr, rho=0.05, eps=1e-6):

    # Base gradient
    loss = exponential_loss_2layer_torch(model, X, y)
    loss.backward()
    g = flatten_grad(model)
    g_norm = g.norm() + 1e-12

    base_weights = [p.detach().clone() for p in model.parameters()]

    # Perturb
    with torch.no_grad():
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.add_(rho * g[idx:idx+n].view_as(p) / g_norm)
            idx += n

    # Gradient at perturbed point
    zero_grad(model)
    loss2 = exponential_loss_2layer_torch(model, X, y)
    loss2.backward()

    g2 = flatten_grad(model)
    F_diag = torch.abs(g2) + eps
    nat_grad = g2 / F_diag

    # Restore and apply NGD update
    with torch.no_grad():
        idx = 0
        for p, base in zip(model.parameters(), base_weights):
            n = p.numel()
            p.copy_(base)
            p -= lr * nat_grad[idx:idx+n].view_as(p)
            idx += n

    zero_grad(model)
    return model
