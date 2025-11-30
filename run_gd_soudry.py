OUTPUT_DIR = "soudry"

from regression.dataset import make_soudry_dataset
from regression.metrics import get_empirical_max_margin, get_angle
from regression.optimizers import (
    step_gd, step_sam_stable, step_ngd_stable, step_sam_ngd_stable
)
from regression.trainer import run_training
from regression.plotting import plot_all

SEED = 42
import numpy as np
import random
np.random.seed(SEED)
random.seed(SEED)

def main():
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, margin=0.1, sigma=3.0)
    w_star = get_empirical_max_margin(X, y)

    print("Angle(v, w*):", get_angle(v_pop, w_star))

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]

    optimizers = {
        "GD": step_gd,
        "SAM": step_sam_stable,
        "NGD": step_ngd_stable,
        "SAM_NGD": step_sam_ngd_stable
    }

    results = run_training(X, y, w_star, optimizers, learning_rates)
    plot_all(results, learning_rates, optimizers , OUTPUT_DIR)

if __name__ == "__main__":
    main()

