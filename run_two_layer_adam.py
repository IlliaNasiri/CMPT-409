# ===============================================================
# run_adam_experiments.py   (Adam / Adagrad / SAM-Adam / SAM-Adagrad)
# ===============================================================
import numpy as np
from linearLayer.dataset import make_soudry_dataset
from linearLayer.trainer import run_training
from linearLayer.plotting import plot_all

from linearLayer.adam_optimizers import (
    make_torch_adam_step,
    make_torch_adagrad_step,
    make_torch_sam_adam_step,
    make_torch_sam_adagrad_step
)


def main():
    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(N, D)
    print("Dataset ready:", X.shape, y.shape)

    # ----------------------------------------------------------
    # Model width
    # ----------------------------------------------------------
    k = 50

    # ----------------------------------------------------------
    # Learning rate sweep
    # ----------------------------------------------------------
    learning_rates = [1e-4, 3e-4, 1e-3]
    total_iters = 100_000

    # ----------------------------------------------------------
    # REGISTER ALL TORCH OPTIMIZERS FOR ALL LRs
    # ----------------------------------------------------------
    optimizers = {}

    for lr in learning_rates:

        optimizers[f"Adam_lr{lr}"] = make_torch_adam_step(
            (k, D), (10, k), lr
        )

        optimizers[f"Adagrad_lr{lr}"] = make_torch_adagrad_step(
            (k, D), (10, k), lr
        )

        optimizers[f"SAM_Adam_lr{lr}"] = make_torch_sam_adam_step(
            (k, D), (10, k), lr
        )

        optimizers[f"SAM_Adagrad_lr{lr}"] = make_torch_sam_adagrad_step(
            (k, D), (10, k), lr
        )

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    results = run_training(
        X, y,
        optimizers=optimizers,
        learning_rates=learning_rates,  # still needed for plotting
        k=k,
        total_iters=total_iters,
        debug=True
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="adam_family"
    )


if __name__ == "__main__":
    main()
