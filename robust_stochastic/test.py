import numpy as np
import matplotlib.pyplot as plt

def plot_grid_from_npz(
    npz_path,
    learning_rates,
    rhos,
    metric_key="angle",           # "angle", "dist", "loss", or "err"
    metric_name="Angle (radians)",
    save_path="grid_angle.png",
):
    """
    Make a |rhos| x |learning_rates| grid of plots.

    Each subplot shows: SGD, SGD+SAM, NGD, NGD+SAM at that (lr, rho).

    The NPZ is assumed to come from plotting.plot_all(...),
    where keys look like:
        lr{lr}_SGD_steps,         lr{lr}_SGD_angle, ...
        lr{lr}_SGD+SAM_rho={r}_steps, ...
        lr{lr}_NGD_steps,         lr{lr}_NGD+SAM_rho={r}_angle, ...

    """
    data = np.load(npz_path)

    n_rows = len(rhos)
    n_cols = len(learning_rates)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True,
        sharey=True,
    )

    # Make sure axes is 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Colors / labels for the 4 methods we want in each subplot
    methods = [
        ("SGD",               "SGD"),
        ("SGD+SAM_rho={rho}", "SGD+SAM"),
        ("NGD",               "NGD"),
        ("NGD+SAM_rho={rho}", "NGD+SAM"),
    ]

    for i, rho in enumerate(rhos):
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]

            # column titles (top row)
            if i == 0:
                ax.set_title(f"LR = {lr}")

            # row labels (leftmost column)
            if j == 0:
                ax.set_ylabel(f"ρ = {rho}")

            for key_template, nice_label in methods:
                opt_name = key_template.format(rho=rho)
                base = f"lr{lr}_{opt_name}"

                steps_key  = f"{base}_steps"
                metric_arr_key = f"{base}_{metric_key}"

                # Some combos might not exist; just skip them
                if steps_key not in data or metric_arr_key not in data:
                    continue

                steps = data[steps_key]
                ys    = data[metric_arr_key]  # +1e-12 to avoid log(0)

                ax.plot(steps, ys, label=nice_label)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)

            if i == n_rows - 1:
                ax.set_xlabel("epoch")

    # One global legend (use last axis' handles)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10)

    fig.suptitle(metric_name, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved grid plot → {save_path}")


# if __name__ == "__main__":
#     # Adjust this path to your actual results.npz
#     npz_path = "experiments/soudry_sam_rho_sweep/Arya_Experiments/results.npz"

#     # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
#     learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
#     # rhos = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0]
#     rhos = [0.05, 0.1, 1.0, 5.0, 15.0]

#     # Example: angle grid
#     plot_grid_from_npz(
#         npz_path,
#         learning_rates,
#         rhos,
#         metric_key="err",
#         metric_name="Test Error Rate",
#         save_path="grid_test_error.png",
#     )

#     # You can similarly call with metric_key="dist", "loss", or "err"


# --------------------------------------------------------------------
# NEW: helper to plot a single curve (one optimizer, one lr, one metric)
# --------------------------------------------------------------------
def plot_single_from_npz(
    npz_path,
    lr,
    opt_name,
    metric_key,
    metric_name,
    save_path,
):
    """
    Plot a single curve from results.npz:
      - lr: float learning rate used in training (must match what was used in run_experiments.py)
      - opt_name: e.g. "SGD", "SGD+SAM_rho=1.0", "NGD", "NGD+SAM_rho=1.0"
      - metric_key: "err", "angle", "dist", or "loss"
    """
    data = np.load(npz_path)

    base = f"lr{lr}_{opt_name}"
    steps_key  = f"{base}_steps"
    metric_arr_key = f"{base}_{metric_key}"

    if steps_key not in data or metric_arr_key not in data:
        raise KeyError(f"Missing keys in NPZ: {steps_key}, {metric_arr_key}")

    steps = data[steps_key]
    ys    = data[metric_arr_key] # + 1e-12

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(steps, ys)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}\n{opt_name}, lr={lr}")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot → {save_path}")


if __name__ == "__main__":
    # Adjust this path to your actual results.npz
    npz_path = "experiments/fresh_run_logistic/2025-12-13_10-27-49/results.npz"

    # learning_rates used in run_experiments.py
    # learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    # rhos = [0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 50.0]
    rhos = [0.05, 0.1, 1.0, 5.0, 15.0]

    # Example: grid over (lr, rho) for some metric
    plot_grid_from_npz(
        npz_path,
        learning_rates,
        rhos,
        metric_key="loss",
        metric_name="Logistic Loss",
        save_path="loss_grid_logistic.png",
    )

    # ------------------------------------------------------------
    # NEW: individual plots requested
    # ------------------------------------------------------------
    # 1) Classification TEST ERROR
    # plot_single_from_npz(
    #     npz_path,
    #     lr=1e-3,
    #     opt_name="SGD",
    #     metric_key="err",
    #     metric_name="Classification Test Error",
    #     save_path="sgd_lr0.001_test_error.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1.0,
    #     opt_name="SGD+SAM_rho=1.0",
    #     metric_key="err",
    #     metric_name="Classification Test Error",
    #     save_path="sgd_sam_rho1.0_lr1.0_test_error.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1e-1,
    #     opt_name="NGD",
    #     metric_key="err",
    #     metric_name="Classification Test Error",
    #     save_path="ngd_lr0.1_test_error.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1.0,
    #     opt_name="NGD+SAM_rho=1.0",
    #     metric_key="err",
    #     metric_name="Classification Test Error",
    #     save_path="ngd_sam_rho1.0_lr1.0_test_error.png",
    # )

    # # 2) Angle from max-margin solution
    # plot_single_from_npz(
    #     npz_path,
    #     lr=1e-3,
    #     opt_name="SGD",
    #     metric_key="angle",
    #     metric_name="Angle from Max-Margin Direction (radians)",
    #     save_path="sgd_lr0.001_angle.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1.0,
    #     opt_name="SGD+SAM_rho=1.0",
    #     metric_key="angle",
    #     metric_name="Angle from Max-Margin Direction (radians)",
    #     save_path="sgd_sam_rho1.0_lr1.0_angle.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1e-1,
    #     opt_name="NGD",
    #     metric_key="angle",
    #     metric_name="Angle from Max-Margin Direction (radians)",
    #     save_path="ngd_lr0.1_angle.png",
    # )

    # plot_single_from_npz(
    #     npz_path,
    #     lr=1.0,
    #     opt_name="NGD+SAM_rho=1.0",
    #     metric_key="angle",
    #     metric_name="Angle from Max-Margin Direction (radians)",
    #     save_path="ngd_sam_rho1.0_lr1.0_angle.png",
    # )
