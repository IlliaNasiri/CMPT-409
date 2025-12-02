import numpy as np
from tqdm import tqdm
from plotting import plot_all

from metrics import (
    angle_between,
    direction_distance,
    get_error_rate,
)
from optimizers import (
    exponential_loss,
    stochastic_exponential_grad,
    sgd_step,
    sgd_sam_step,
    stochastic_ngd_step,
    sam_ngd_step
)


def make_algorithms(
    rhos,
    include_sgd=True,
    include_ngd=True,
):
    """
    Return a dict mapping algorithm name -> update function.

    Each update function has signature:
        w_new = step_fn(w, X, y, batch_indices, lr)

    rhos : iterable of SAM radii to test
    """

    algos = {}

    # -------------------------
    # Plain SGD
    # -------------------------
    if include_sgd:
        def sgd_update(w, X, y, batch_indices, lr):
            grad = stochastic_exponential_grad(w, batch_indices, X, y)
            return sgd_step(w, grad, lr)

        algos["SGD"] = sgd_update

        def make_sgd_sam_update(rho):
            def sgd_sam_update(w, X, y, batch_indices, lr):
                sgrad = stochastic_exponential_grad(w, batch_indices, X, y)
                return sgd_sam_step(w, sgrad, batch_indices, X, y, lr, rho=rho)
            return sgd_sam_update

        for rho in rhos:
            name = f"SGD+SAM_rho={rho}"
            algos[name] = make_sgd_sam_update(rho)

    # -------------------------
    # NGD
    # -------------------------
    if include_ngd:
        def ngd_update(w, X, y, batch_indices, lr):
            sgrad = stochastic_exponential_grad(w, batch_indices, X, y)
            return stochastic_ngd_step(w, sgrad, batch_indices, X, y, lr)

        algos["NGD"] = ngd_update

        def make_ngd_sam_update(rho):
            def ngd_sam_update(w, X, y, batch_indices, lr):
                sgrad = stochastic_exponential_grad(w, batch_indices, X, y)
                return sam_ngd_step(w, sgrad, batch_indices, X, y, lr, rho=rho) 
            return ngd_sam_update

        for rho in rhos:
            name = f"NGD+SAM_rho={rho}"
            algos[name] = make_ngd_sam_update(rho)

    return algos

def run_experiments(
    X,
    y,
    X_test,
    y_test,
    w_star,
    algorithms,      # dict: name -> step_fn(w, X, y, batch_indices, lr)
    learning_rates,
    batch_size,
    num_epochs,
    debug=True,
):
    """
    Run experiments for a list of algorithms over a sweep of learning rates.

    Parameters
    ----------
    X, y : data
    w_star : ground-truth max-margin direction
    algorithms : dict
        {"SGD": step_fn, "SGD+SAM": step_fn, ...}
    learning_rates : list of float
    batch_size : int
    num_epochs : int

    Returns
    -------
    results : dict
        results[lr][algo_name] = {
            "steps": [epochs_logged],
            "norm":  [||w_t||_2],
            "loss":  [exponential_loss(w_t)],
            "angle": [angle_between(w_t, w_star)],
            "dist":  [direction_distance(w_t, w_star)],
            "err":   [error_rate(w_t)],
        }
    """

    n, d = X.shape
    steps_per_epoch = n // batch_size
    total_steps = num_epochs * steps_per_epoch

    # ALWAYS include step=1; log-spaced thresholds in *iteration* count
    record_steps = np.unique(
        np.concatenate(([1], np.logspace(0, np.log10(total_steps), 400))).astype(int)
    )
    if debug:
        print("\n[DEBUG] record_steps[:10] =", record_steps[:10])

    results = {}

    # ==========================================================
    # LOOP OVER LEARNING RATES
    # ==========================================================
    for lr in learning_rates:
        if debug:
            print(f"\n=== Learning Rate Sweep: lr={lr} ===")

        # one parameter vector per algorithm
        ws = {name: np.zeros(d) for name in algorithms.keys()}

        # histories per algorithm
        hists = {
            name: {
                "steps": [],
                "norm": [],
                "loss": [],
                "angle": [],
                "dist": [],
                "err": [],
            }
            for name in algorithms.keys()
        }

        t_global = 0         # counts SGD updates
        rec_idx = 0          # index into record_steps

        # ------------------------------------------------------
        # MAIN TRAINING LOOP: case 2 (sampling without replacement)
        # ------------------------------------------------------
        for epoch in tqdm(range(num_epochs), desc=f"Epochs (no-replacement), lr={lr}"):
            # shuffle once per epoch, partition into mini-batches
            perm = np.random.permutation(n)

            for j in range(steps_per_epoch):
                t_global += 1
                batch_indices = perm[j * batch_size : (j + 1) * batch_size]

                # UPDATE ALL REQUESTED ALGORITHMS
                for name, step_fn in algorithms.items():
                    w_old = ws[name]
                    w_new = step_fn(w_old, X, y, batch_indices, lr)

                    # (optional) numeric safety check
                    if np.isnan(w_new).any() or np.isinf(w_new).any():
                        raise FloatingPointError(
                            f"NaN/Inf in '{name}' at step {t_global} (lr={lr})"
                        )
                    ws[name] = w_new

                # LOG WHEN WE HIT A THRESHOLD
                while rec_idx < len(record_steps) and t_global >= record_steps[rec_idx]:
                    epoch_x = t_global / steps_per_epoch  # x-axis in *epochs*

                    for name in algorithms.keys():
                        wcur = ws[name]
                        hist = hists[name]

                        hist["steps"].append(epoch_x)
                        hist["norm"].append(np.linalg.norm(wcur))
                        hist["loss"].append(exponential_loss(wcur, X, y))
                        hist["angle"].append(angle_between(wcur, w_star))
                        hist["dist"].append(direction_distance(wcur, w_star))
                        hist["err"].append(get_error_rate(wcur, X_test, y_test))

                    rec_idx += 1

                if rec_idx >= len(record_steps):
                    break

            if rec_idx >= len(record_steps):
                break

        if debug:
            for name in algorithms.keys():
                print(
                    f"[DEBUG] lr={lr}, algo={name}, "
                    f"logged {len(hists[name]['steps'])} points."
                )

        results[lr] = hists

    return results


if __name__ == "__main__":
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test  = np.load("X_test.npy")
    y_test  = np.load("y_test.npy")
    w_star  = np.load("w_star.npy")

    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    # learning_rates = [0.01]   # fixed LR, based on empirical performance
    rhos = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0]
    # rhos = [0.1]    # single rho, based on empirical performance

    algorithms = make_algorithms(rhos, include_sgd=True, include_ngd=True)  

    results = run_experiments(
        X_train,
        y_train,
        X_test,
        y_test,
        w_star,
        algorithms=algorithms,
        learning_rates=learning_rates,
        batch_size=128,
        num_epochs=10000,
        debug=True,
    )

    plot_all(
        results=results,
        learning_rates=learning_rates,
        optimizers=list(algorithms.keys()),
        experiment_name="soudry_sam_rho_sweep",
    )


