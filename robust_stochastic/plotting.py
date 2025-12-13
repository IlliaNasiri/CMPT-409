import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def plot_all(
    results,
    learning_rates,
    optimizers,
    experiment_name="default",
    save_combined=True,
    save_separate=True,
    post_training=True     # <--- ENABLE THIS TO SAVE .NPZ
):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join("experiments", experiment_name, timestamp)
    os.makedirs(base_dir, exist_ok=True)

    print(f"\n[Saving results to]: {base_dir}\n")

    # ------------------------------
    # METRICS DEFINITION
    # ------------------------------
    metrics = [
        ("norm", "Weight Norm", "norm.png"),
        ("dist", "Direction Distance", "distance.png"),
        ("angle", "Angle (radians)", "angle.png"),
        ("loss", "Exponential Loss", "loss.png"),
        ("err",  "Classification Error", "error.png"),
    ]

    # ============================================================
    # 🔥 0. SAVE RESULTS AS NPZ BEFORE PLOTTING
    # ============================================================
    if post_training:
        npz_path = os.path.join(base_dir, "results.npz")

        flat = {}

        for lr, sub in results.items():
            for opt_name, hist in sub.items():
                prefix = f"lr{lr}_{opt_name}"
                flat[f"{prefix}_steps"] = np.array(hist["steps"])
                flat[f"{prefix}_norm"]  = np.array(hist["norm"])
                flat[f"{prefix}_dist"]  = np.array(hist["dist"])
                flat[f"{prefix}_angle"] = np.array(hist["angle"])
                flat[f"{prefix}_loss"]  = np.array(hist["loss"])
                flat[f"{prefix}_err"]   = np.array(hist["err"])

        np.savez_compressed(npz_path, **flat)
        print(f"[NumPy] Saved → {npz_path}\n")

    # ============================================================
    # 1. COMBINED PLOTS
    # ============================================================
    if save_combined:
        combined_dir = os.path.join(base_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        for key, title, filename in metrics:

            fig, axes = plt.subplots(1, len(learning_rates), figsize=(18, 5))
            if len(learning_rates) == 1:
                axes = [axes]

            fig.suptitle(title, fontsize=16)

            for i, lr in enumerate(learning_rates):
                ax = axes[i]

                for name in optimizers:
                    hist = results[lr][name]

                    if len(hist["steps"]) > 0:
                        # to prevent log-scale failure for y-axis
                        ys = np.array(hist[key]) + 1e-12

                        # ax.plot(hist["steps"], hist[key], label=name)
                        ax.plot(hist["steps"], ys, label=name)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title(f"LR = {lr}")
                ax.grid(True, which="both", alpha=0.3)
                if i == 0:
                    ax.legend()

            out_path = os.path.join(combined_dir, filename)
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved combined plot → {out_path}")

    # ============================================================
    # 2. SEPARATE OPTIMIZER PLOTS
    # ============================================================
    if save_separate:
        separate_dir = os.path.join(base_dir, "separate")
        os.makedirs(separate_dir, exist_ok=True)

        for name in optimizers:
            opt_dir = os.path.join(separate_dir, name)
            os.makedirs(opt_dir, exist_ok=True)

            for key, title, filename in metrics:

                fig, ax = plt.subplots(figsize=(6, 5))
                fig.suptitle(f"{title} — {name}", fontsize=14)

                for lr in learning_rates:
                    hist = results[lr][name]
                    if len(hist["steps"]) > 0:
                        ys = np.array(hist[key]) + 1e-12
                        # ax.plot(hist["steps"], hist[key]+1e-12,
                        #         label=f"LR={lr}")
                        ax.plot(hist["steps"], ys,
                                label=f"LR={lr}")

                ax.set_xscale("log")
                ax.set_yscale("log")

                ax.grid(True, which="both", alpha=0.3)
                ax.legend()

                out_path = os.path.join(opt_dir, filename)
                plt.savefig(out_path)
                plt.close(fig)
                print(f"Saved separate plot → {out_path}")

    return base_dir
