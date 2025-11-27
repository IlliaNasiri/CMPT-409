import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===============================================================
# Data generation
# ===============================================================

def make_soudry_dataset(n=200, d=5000, margin=0.1, sigma=3.0):
    v = np.ones(d) / np.sqrt(d)
    n2 = n // 2

    X_pos = margin * v + sigma * np.random.randn(n2, d)
    X_neg = -margin * v + sigma * np.random.randn(n2, d)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n2), -np.ones(n2)])

    perm = np.random.permutation(n)
    return X[perm], y[perm], v


# ===============================================================
# Logistic utilities
# ===============================================================

def logistic_loss(w, X, y):
    margins = y * (X @ w)
    return np.mean(np.logaddexp(0, -margins))

def logistic_grad(w, X, y):
    margins = y * (X @ w)
    probs = 1.0 / (1.0 + np.exp(margins))
    return -(y * probs) @ X / len(X)

def angle_between(u, v):
    dot = np.dot(u, v)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    val = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(val)

def direction_distance(u, v):
    u_hat = u / np.linalg.norm(u)
    v_hat = v / np.linalg.norm(v)
    return np.linalg.norm(u_hat - v_hat)


# ===============================================================
# Optimization rules
# ===============================================================

def gd_step(w, grad, lr):
    return w - lr * grad

def ngd_step(w, grad, lr):
    gnorm = np.linalg.norm(grad) + 1e-12
    return w - lr * (grad / gnorm)

def sam_step(w, grad, X, y, lr, rho):
    gnorm = np.linalg.norm(grad) + 1e-12
    eps = rho * grad / gnorm
    grad2 = logistic_grad(w + eps, X, y)
    return w - lr * grad2

def sam_ngd_step(w, grad, X, y, lr, rho):
    gnorm = np.linalg.norm(grad) + 1e-12
    eps = rho * grad / gnorm
    grad2 = logistic_grad(w + eps, X, y)
    gnorm2 = np.linalg.norm(grad2) + 1e-12
    return w - lr * (grad2 / gnorm2)