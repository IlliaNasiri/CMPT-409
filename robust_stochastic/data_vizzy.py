import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

def project_onto_wstar_plane(X, w_star, rng=None):
    """
    Project X onto the 2D plane spanned by:
        e1 = w_star / ||w_star||
        e2 = some unit vector orthogonal to w_star

    Returns
    -------
    X_proj : (n, 2) array
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Ensure unit w_star
    e1 = w_star / (np.linalg.norm(w_star) + 1e-12)

    # Sample random vector and make it orthogonal to w_star
    v = rng.normal(size=w_star.shape)
    v -= np.dot(v, e1) * e1
    e2 = v / (np.linalg.norm(v) + 1e-12)

    # Coordinates: [X·e1, X·e2]
    coord1 = X @ e1
    coord2 = X @ e2
    X_proj = np.stack([coord1, coord2], axis=1)
    return X_proj


def plot_train_test_data(X_train, y_train, X_test, y_test, w_star, max_points=3000):
    """
    Visualize training and test data in 2D by projecting onto the
    plane spanned by w_star and an orthogonal direction.
    """

    rng = np.random.default_rng(0)

    # Optionally subsample if sets are huge for nicer plotting
    def maybe_subsample(X, y, max_points):
        n = X.shape[0]
        if n <= max_points:
            return X, y
        idx = rng.choice(n, size=max_points, replace=False)
        return X[idx], y[idx]

    X_tr_vis, y_tr_vis = maybe_subsample(X_train, y_train, max_points)
    X_te_vis, y_te_vis = maybe_subsample(X_test, y_test, max_points)

    # Use the SAME basis (w_star plane) for both train and test
    X_tr_proj = project_onto_wstar_plane(X_tr_vis, w_star, rng)
    X_te_proj = project_onto_wstar_plane(X_te_vis, w_star, rng)

    # Masks for classes
    tr_pos = (y_tr_vis == 1)
    tr_neg = (y_tr_vis == -1)
    te_pos = (y_te_vis == 1)
    te_neg = (y_te_vis == -1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_tr, ax_te = axes

    # -------------------------
    # Train plot
    # -------------------------
    ax_tr.scatter(
        X_tr_proj[tr_pos, 0], X_tr_proj[tr_pos, 1],
        alpha=0.6, s=20, label="Train +1", marker="o"
    )
    ax_tr.scatter(
        X_tr_proj[tr_neg, 0], X_tr_proj[tr_neg, 1],
        alpha=0.6, s=20, label="Train -1", marker="x"
    )
    ax_tr.set_title("Training Data (projected onto w* plane)")
    ax_tr.set_xlabel(r"Coordinate along $w_\star$")
    ax_tr.set_ylabel(r"Orthogonal coordinate")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend()

    # -------------------------
    # Test plot
    # -------------------------
    ax_te.scatter(
        X_te_proj[te_pos, 0], X_te_proj[te_pos, 1],
        alpha=0.6, s=20, label="Test +1", marker="o"
    )
    ax_te.scatter(
        X_te_proj[te_neg, 0], X_te_proj[te_neg, 1],
        alpha=0.6, s=20, label="Test -1", marker="x"
    )
    ax_te.set_title("Test Data (projected onto w* plane)")
    ax_te.set_xlabel(r"Coordinate along $w_\star$")
    ax_te.set_ylabel(r"Orthogonal coordinate")
    ax_te.grid(True, alpha=0.3)
    ax_te.legend()

    plt.tight_layout()
    plt.show()


def get_empirical_max_margin(X, y):
    clf = LinearSVC(C=1e2, fit_intercept=False, dual="auto", max_iter=20000, tol=1e-6)
    clf.fit(X, y)
    w_svm = clf.coef_.flatten()

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    print(f"SVM Separation Accuracy: {acc * 100:.2f}%")
    if acc < 1.0:
        print("[WARNING] Data not perfectly separable by SVM!")
    return w_svm / np.linalg.norm(w_svm)

# Example usage (assuming you've already loaded them):
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")
w_star  = np.load("w_star.npy")

# plot_train_test_data(X_train, y_train, X_test, y_test, w_star)


# w_star = get_empirical_max_margin(X_train, y_train)

# np.save("w_star.npy", w_star)

print(np.linalg.norm(w_star))

# print(X_train.shape)
