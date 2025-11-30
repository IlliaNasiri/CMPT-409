import numpy as np

def forward_scores(W1, W2, X):
    Z = W1 @ X.T
    pred = W2 @ Z
    f = pred.mean(axis=0)
    return f

def exponential_loss_2layer(W1, W2, X, y):
    f = forward_scores(W1, W2, X)
    margins = y * f
    safe = np.clip(margins, -50, 50)
    return np.mean(np.exp(-safe))

def error_rate_2layer(W1, W2, X, y):
    f = forward_scores(W1, W2, X)
    preds = np.sign(f)
    return np.mean(preds != y)
