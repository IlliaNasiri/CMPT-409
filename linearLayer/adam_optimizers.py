import torch

# ============================================================
# Forward pass for 2-layer network
# ============================================================
def torch_forward(W1, W2, X):
    # W1: (k, D)
    # W2: (10, k)
    # X : (N, D)
    Z = W1 @ X.T        # (k, N)
    pred = W2 @ Z       # (10, N)
    f = pred.mean(dim=0)  # (N,)
    return Z, pred, f


# ============================================================
# SEPARATE FUNCTION FOR COMPUTING EXP LOSS
# ============================================================
def exponential_loss_torch(W1, W2, X, y):
    _, _, f = torch_forward(W1, W2, X)
    margins = y * f
    return torch.mean(torch.exp(-torch.clamp(margins, min=-50)))


# ============================================================
# UTIL: load numpy weights into persistent torch params
# ============================================================
def _load_numpy_weights(W1_t, W2_t, W1_np, W2_np, device):
    with torch.no_grad():
        W1_t.copy_(torch.tensor(W1_np, dtype=W1_t.dtype, device=device))
        W2_t.copy_(torch.tensor(W2_np, dtype=W2_t.dtype, device=device))


# ============================================================
# 1) PURE ADAM
# ============================================================
def make_torch_adam_step(W1_shape, W2_shape, lr, device="cpu"):

    W1_t = torch.zeros(W1_shape, dtype=torch.float32, requires_grad=True, device=device)
    W2_t = torch.zeros(W2_shape, dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam([W1_t, W2_t], lr=lr)

    def step(W1_np, W2_np, X_np, y_np, _unused=None):
        _load_numpy_weights(W1_t, W2_t, W1_np, W2_np, device)

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        loss = exponential_loss_torch(W1_t, W2_t, X, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        return W1_t.detach().cpu().numpy(), W2_t.detach().cpu().numpy()

    return step


# ============================================================
# 2) PURE ADAGRAD
# ============================================================
def make_torch_adagrad_step(W1_shape, W2_shape, lr, eps=1e-8, device="cpu"):

    W1_t = torch.zeros(W1_shape, dtype=torch.float32, requires_grad=True, device=device)
    W2_t = torch.zeros(W2_shape, dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adagrad([W1_t, W2_t], lr=lr, eps=eps)

    def step(W1_np, W2_np, X_np, y_np, _unused=None):
        _load_numpy_weights(W1_t, W2_t, W1_np, W2_np, device)

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        loss = exponential_loss_torch(W1_t, W2_t, X, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        return W1_t.detach().cpu().numpy(), W2_t.detach().cpu().numpy()

    return step


def zero_grad_params(params):
    for p in params:
        p.grad = None


# ------------------------------------------------------------
# SAM + ADAM  (correct)
# ------------------------------------------------------------
def make_torch_sam_adam_step(D, lr, rho=0.05, device="cpu"):
    """
    Correct SAM-Adam:
        1) grad at w
        2) perturb → w_adv
        3) grad at w_adv
        4) restore w and take Adam step using grad_adv
    """

    # persistent parameter + state
    w_t = torch.zeros(D, dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam([w_t], lr=lr)

    def step(w_np, X_np, y_np, _unused=None):
        nonlocal w_t, opt

        # load weights
        with torch.no_grad():
            w_t.copy_(torch.tensor(w_np, device=device))

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        # ------------------------------
        # 1) gradient at w
        # ------------------------------
        w = w_t.clone().detach().requires_grad_(True)

        margins = y * (X @ w)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))

        grad = torch.autograd.grad(loss, w)[0]
        g_norm = grad.norm() + 1e-12

        # Save original w
        base_w = w.detach().clone()

        # ------------------------------
        # 2) perturb: w_adv
        # ------------------------------
        w_adv = (w + rho * grad / g_norm).detach()

        # ------------------------------
        # 3) gradient at w_adv
        # ------------------------------
        w_adv_t = w_adv.clone().detach().requires_grad_(True)

        margins_adv = y * (X @ w_adv_t)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))
        grad_adv = torch.autograd.grad(loss_adv, w_adv_t)[0]

        # ------------------------------
        # 4) restore w and Adam update
        # ------------------------------
        with torch.no_grad():
            w_t.copy_(base_w)

        w_t.requires_grad_(True)
        w_t.grad = grad_adv.clone()

        opt.step()

        return w_t.detach().cpu().numpy()

    return step


# ------------------------------------------------------------
# SAM + ADAGRAD  (correct)
# ------------------------------------------------------------
def make_torch_sam_adagrad_step(D, lr, rho=0.05, eps=1e-8, device="cpu"):
    """
    Correct SAM-Adagrad (same SAM steps as above)
    """

    w_t = torch.zeros(D, dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adagrad([w_t], lr=lr, eps=eps)

    def step(w_np, X_np, y_np, _unused=None):
        nonlocal w_t, opt

        # load weights
        with torch.no_grad():
            w_t.copy_(torch.tensor(w_np, device=device))

        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.float32, device=device)

        # ------------------------------
        # 1) gradient at w
        # ------------------------------
        w = w_t.clone().detach().requires_grad_(True)

        margins = y * (X @ w)
        loss = torch.mean(torch.exp(-torch.clamp(margins, min=-50)))
        grad = torch.autograd.grad(loss, w)[0]

        g_norm = grad.norm() + 1e-12
        base_w = w.detach().clone()

        # ------------------------------
        # 2) perturb
        # ------------------------------
        w_adv = (w + rho * grad / g_norm).detach()

        # ------------------------------
        # 3) gradient at perturbed weights
        # ------------------------------
        w_adv_t = w_adv.clone().detach().requires_grad_(True)

        margins_adv = y * (X @ w_adv_t)
        loss_adv = torch.mean(torch.exp(-torch.clamp(margins_adv, min=-50)))
        grad_adv = torch.autograd.grad(loss_adv, w_adv_t)[0]

        # ------------------------------
        # 4) restore w and Adagrad update
        # ------------------------------
        with torch.no_grad():
            w_t.copy_(base_w)

        w_t.requires_grad_(True)
        w_t.grad = grad_adv.clone()

        opt.step()

        return w_t.detach().cpu().numpy()

    return step
