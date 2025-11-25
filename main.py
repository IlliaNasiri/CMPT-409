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


# ===============================================================
# Training loop for all optimizers
# ===============================================================

def main():
    print("Generating dataset...")
    X, y, w_star = make_soudry_dataset(
        n=200,
        d=5000,
        margin=0.1,
        sigma=3.0,
    )

    sigma_max = np.linalg.norm(X, ord=2)
    lr = 1.0 / (sigma_max ** 2)
    rho = 0.1
    print("Learning rate:", lr)

    names = ["GD", "NGD", "SAM", "SAM_NGD"]

    steps_dict = {k: [] for k in names}
    norms = {k: [] for k in names}
    losses = {k: [] for k in names}
    angles = {k: [] for k in names}
    dists = {k: [] for k in names}

    ws = {
        "GD": np.zeros(X.shape[1]),
        "NGD": np.zeros(X.shape[1]),
        "SAM": np.zeros(X.shape[1]),
        "SAM_NGD": np.zeros(X.shape[1]),
    }

    total_iters = 300000
    record_steps = np.unique(np.logspace(0, np.log10(total_iters), 400).astype(int))
    step_idx = 0

    print("Training...")

    for t in tqdm(range(1, total_iters + 1)):

        # compute each optimizer's gradient from its own weights
        grad_GD       = logistic_grad(ws["GD"], X, y)
        grad_NGD      = logistic_grad(ws["NGD"], X, y)
        grad_SAM      = logistic_grad(ws["SAM"], X, y)
        grad_SAM_NGD  = logistic_grad(ws["SAM_NGD"], X, y)

        # update all weights with their own rule
        ws["GD"]       = gd_step(ws["GD"], grad_GD, lr)
        ws["NGD"]      = ngd_step(ws["NGD"], grad_NGD, lr)
        ws["SAM"]      = sam_step(ws["SAM"], grad_SAM, X, y, lr, rho)
        ws["SAM_NGD"]  = sam_ngd_step(ws["SAM_NGD"], grad_SAM_NGD, X, y, lr, rho)

        if t == record_steps[step_idx]:
            for name in names:
                wcur = ws[name]
                steps_dict[name].append(t)
                norms[name].append(np.linalg.norm(wcur))
                losses[name].append(logistic_loss(wcur, X, y))
                angles[name].append(angle_between(wcur, w_star))
                dists[name].append(direction_distance(wcur, w_star))

            step_idx += 1
            if step_idx >= len(record_steps):
                break

    # ===========================================================
    # Plots
    # ===========================================================

    # Norm plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], norms[name], label=name)
    plt.xscale("log")
    plt.title("||w(t)|| growth")
    plt.xlabel("iteration")
    plt.ylabel("norm")
    plt.legend()
    plt.grid()
    plt.savefig("norm_compare.png")

    # Loss plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], losses[name], label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Logistic loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.savefig("loss_compare.png")

    # Angle plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], angles[name], label=name)
    plt.xscale("log")
    plt.title("Angle between w(t) and w*")
    plt.xlabel("iteration")
    plt.ylabel("angle (radians)")
    plt.legend()
    plt.grid()
    plt.savefig("angle_compare.png")

    # Distance plot
    plt.figure()
    for name in names:
        plt.plot(steps_dict[name], dists[name], label=name)
    plt.xscale("log")
    plt.title("Direction distance")
    plt.xlabel("iteration")
    plt.ylabel("||w_hat - w_star_hat||")
    plt.legend()
    plt.grid()
    plt.savefig("distance_compare.png")

    print("Done. Saved norm_compare.png, loss_compare.png, angle_compare.png, distance_compare.png")


if __name__ == "__main__":
    main()

