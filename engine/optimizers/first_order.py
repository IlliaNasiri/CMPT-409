import numpy as np

def step_gd(w, X, y, lr):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    return w - lr * grad

def step_ngd_stable(w, X, y, lr):
    margins = y * (X @ w)
    neg_margins = -margins
    shift = np.max(neg_margins)
    exps = np.exp(neg_margins - shift)
    softmax_weights = exps / np.sum(exps)

    direction = - (X.T @ (y * softmax_weights))
    return w - lr * direction

def step_sam_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    gnorm = np.linalg.norm(grad) + 1e-12

    eps = rho * grad / gnorm
    w_adv = w + eps

    margins_adv = y * (X @ w_adv)
    safe_margins_adv = np.clip(margins_adv, -50, None)
    coeffs_adv = np.exp(-safe_margins_adv)
    grad_adv = - (X.T @ (y * coeffs_adv)) / len(y)

    return w - lr * grad_adv

def step_sam_ngd_stable(w, X, y, lr, rho=0.05):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -50, None)
    coeffs = np.exp(-safe_margins)
    grad = - (X.T @ (y * coeffs)) / len(y)
    gnorm = np.linalg.norm(grad) + 1e-12

    eps = rho * grad / gnorm
    w_adv = w + eps

    return step_ngd_stable(w_adv, X, y, lr)
