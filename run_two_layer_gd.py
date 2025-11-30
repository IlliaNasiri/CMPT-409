# ===============================================================
# run_ngd_experiments.py   (GD / NGD / SAM / SAM-NGD)
# ===============================================================
import numpy as np
from linearLayer.dataset import make_soudry_dataset
from linearLayer.trainer import run_training
from linearLayer.plotting import plot_all
from linearLayer.optimizers import (
    step_gd, step_ngd, step_sam, step_sam_ngd
)


def main():
    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(N, D)
    print("Dataset ready:", X.shape, y.shape)

    # ----------------------------------------------------------
    # 2-layer width
    # ----------------------------------------------------------
    k = 50

    # ----------------------------------------------------------
    # Learning rate sweep
    # ----------------------------------------------------------
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    total_iters = 100_000

    # ----------------------------------------------------------
    # Register NumPy optimizers
    # ----------------------------------------------------------
    optimizers = {
        "GD": step_gd,
        "NGD": step_ngd,
        "SAM": step_sam,
        "SAM_NGD": step_sam_ngd,
    }

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    results = run_training(
        X, y,
        optimizers=optimizers,
        learning_rates=learning_rates,
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
        experiment_name="2layers_gd_family"
    )


if __name__ == "__main__":
    main()
