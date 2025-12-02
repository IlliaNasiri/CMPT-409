OUTPUT_DIR = "adam/adagrad"

from regression.dataset import make_soudry_dataset
from regression.metrics import get_empirical_max_margin, get_angle
from regression.optimizers import (
    step_gd, step_sam_stable, step_ngd_stable, step_sam_ngd_stable
)

from regression.adam_optimizers import * 

from regression.trainer import run_training
from regression.plotting import plot_all
import numpy as np
import random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def main():
    X, y, v_pop = make_soudry_dataset(n=200, d=5000)
    w_star = get_empirical_max_margin(X, y)

    print("Angle(v, w*):", get_angle(v_pop, w_star))

    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    D = X.shape[1]   # number of features

    optimizers = {

    }


    for lr in learning_rates:
        # --- Standard PyTorch optimizers ---
        optimizers[f"adam_lr={lr}"]          = make_torch_adam_step(D, lr)
        optimizers[f"adagrad_lr={lr}"]       = make_torch_adagrad_step(D, lr)



        # SAM applied to Adam / Adagrad ---
        optimizers[f"sam_adam_lr={lr}"]      = make_torch_sam_adam_step(D, lr)
        optimizers[f"sam_adagrad_lr={lr}"]   = make_torch_sam_adagrad_step(D, lr)
        results = run_training(X, y, w_star, optimizers, learning_rates)
        plot_all(results, learning_rates, optimizers , experiment_name="adam_family")

if __name__ == "__main__":
    main()

