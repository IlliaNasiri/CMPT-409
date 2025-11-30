import numpy as np

def step_gd(w, X, y, lr):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    return w - lr * grad

def step_ngd_stable(w, X, y, lr):
    margins = y * (X @ w)
    shift = np.max(-margins)
    exps = np.exp(-margins - shift)
    softmax_weights = exps / np.sum(exps)

    direction = - (X.T @ (y * softmax_weights))
    return w - lr * direction

def step_sam_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe = np.clip(margins, -50, None)
    coeffs = np.exp(-safe)
    grad = -(X.T @ (y * coeffs)) / len(y)

    gnorm = np.linalg.norm(grad) + 1e-12
    w_adv = w + rho * grad / gnorm

    margins_adv = y * (X @ w_adv)
    safe_adv = np.clip(margins_adv, -50, None)
    coeffs_adv = np.exp(-safe_adv)
    grad_adv = -(X.T @ (y * coeffs_adv)) / len(y)

    return w - lr * grad_adv

def step_sam_ngd_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe = np.clip(margins, -50, None)
    grad = -(X.T @ (y * safe)) / len(y)
    gnorm = np.linalg.norm(grad) + 1e-12
    w_adv = w + rho * grad / gnorm

    return step_ngd_stable(w_adv, X, y, lr)











