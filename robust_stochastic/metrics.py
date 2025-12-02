import numpy as np

def angle_between(u, v):
    dot = np.dot(u, v)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    val = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(val)

def direction_distance(u, v):
    u_hat = u / np.linalg.norm(u)
    v_hat = v / np.linalg.norm(v)
    return np.linalg.norm(u_hat - v_hat)

def get_error_rate(w, X, y):
    preds = np.sign(X@w)
    return np.mean(preds != y)

def compute_gamma_from_direction(X,y,w_dir): # passing w^* gives max margin
    w_dir = np.asarray(w_dir, dtype=float)
    norm_w = np.linalg.norm(w_dir) + 1e-12
    margins = y * (X @ w_dir) / norm_w
    return np.min(margins)

def compute_sigma_max(X, y=None):
    if y is not None:
        X_eff = X * y[:,None]
    else:
        X_eff = X
    
    return np.linalg.norm(X_eff, 2)

def logistic_smoothness_constants(X, y=None):
    n = X.shape[0]
    sigma_max = compute_sigma_max(X,y)
    beta_raw = 0.25 # logistic l''(u) <= 1/4
    L_sum = beta_raw * sigma_max**2
    L_mean = L_sum/n
    beta_mean = beta_raw/n
    return beta_raw, beta_mean, sigma_max, L_sum, L_mean