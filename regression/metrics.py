import numpy as np
from sklearn.svm import LinearSVC

def get_empirical_max_margin(X, y):
    clf = LinearSVC(C=1e6, fit_intercept=False, dual="auto", max_iter=20000, tol=1e-6)
    clf.fit(X, y)
    w = clf.coef_.flatten()
    return w / np.linalg.norm(w)

def get_angle(u, v):
    dot = np.dot(u, v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return np.arccos(np.clip(dot / denom, -1.0, 1.0))

def get_direction_distance(w, w_star):
    return np.linalg.norm(
        w / (np.linalg.norm(w) + 1e-12) -
        w_star / (np.linalg.norm(w_star) + 1e-12)
    )

def exponential_loss(w, X, y):
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -100, 100)
    return np.mean(np.exp(-safe_margins))

def get_error_rate(w, X, y):
    preds = np.sign(X @ w)
    return np.mean(preds != y)
