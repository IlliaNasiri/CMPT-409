import numpy as np

def make_soudry_dataset(n=200, d=5000, margin=0.1, sigma=3.0):
    v = np.ones(d) / np.sqrt(d)
    n2 = n // 2

    noise_pos = sigma * np.random.randn(n2, d)
    noise_neg = sigma * np.random.randn(n2, d)

    X_pos = margin * v + noise_pos
    X_neg = -margin * v + noise_neg

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n2), -np.ones(n2)])

    perm = np.random.permutation(n)
    return X[perm], y[perm], v
