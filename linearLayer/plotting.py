import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_all(results, learning_rates, optimizers, experiment_name="default"):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("experiments", experiment_name, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    metrics = [
        ("loss", "Exponential Loss", "loss.png"),
        ("err",  "Classification Error", "error.png"),
    ]

    # save npz
    flat = {}
    for lr, sub in results.items():
        for opt_name, hist in sub.items():
            prefix = f"lr{lr}_{opt_name}"
            for k in hist:
                flat[f"{prefix}_{k}"] = np.array(hist[k])
    np.savez_compressed(os.path.join(base_dir, "results.npz"), **flat)

    # combined
    combined_dir = os.path.join(base_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    for key, title, fname in metrics:
        fig, axes = plt.subplots(1, len(learning_rates), figsize=(16,5))
        if len(learning_rates) == 1: axes = [axes]

        for i, lr in enumerate(learning_rates):
            ax = axes[i]
            for name in optimizers:
                hist = results[lr][name]
                ax.plot(hist["steps"], hist[key], label=name)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"LR={lr}")
            ax.grid(True)
            if i == 0: ax.legend()

        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(combined_dir, fname))
        plt.close(fig)

    # separate
    separate_dir = os.path.join(base_dir, "separate")
    os.makedirs(separate_dir, exist_ok=True)

    for name in optimizers:
        odir = os.path.join(separate_dir, name)
        os.makedirs(odir, exist_ok=True)

        for key, title, fname in metrics:
            fig, ax = plt.subplots(figsize=(6,5))
            for lr in learning_rates:
                hist = results[lr][name]
                ax.plot(hist["steps"], hist[key], label=f"LR={lr}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend()
            fig.suptitle(title + f" — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(odir, fname))
            plt.close(fig)

    return base_dir
