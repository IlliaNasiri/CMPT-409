import os
import argparse
import numpy as np
from sourdry.plotting import plot_all


def load_results_npz(npz_path):
    """Load NPZ and reconstruct the nested results dict structure:

        results[lr][optimizer] = {
            "steps": [...],
            "dist":  [...],
            "angle": [...],
            "loss":  [...],
            "err":   [...],
        }
    """

    data = np.load(npz_path)

    results = {}

    for key in data.files:
        parts = key.split("_")
        # Example key: lr0.001_GD_steps

        if len(parts) < 3:
            continue

        lr = float(parts[0].replace("lr", ""))
        opt = parts[1]
        metric = parts[2]

        if lr not in results:
            results[lr] = {}
        if opt not in results[lr]:
            results[lr][opt] = {"steps": [], "dist": [], "angle": [], "loss": [], "err": []}

        results[lr][opt][metric] = data[key].tolist()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                        help="Path to experiment folder (contains results.npz)")
    args = parser.parse_args()

    exp_dir = args.exp.rstrip("/")
    npz_path = os.path.join(exp_dir, "results.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Could not find results.npz in: {exp_dir}")

    print(f"Loading NPZ results from:\n  {npz_path}\n")

    # -----------------------------------------------------
    # Load results back into the dict format
    # -----------------------------------------------------
    results = load_results_npz(npz_path)

    # Extract the ordered lists of lrs and optimizers
    learning_rates = sorted(results.keys())
    optimizers = sorted(list(results[learning_rates[0]].keys()))

    print("Loaded learning rates:", learning_rates)
    print("Loaded optimizers:", optimizers)

    # -----------------------------------------------------
    # Run plotting into test/plots
    # -----------------------------------------------------
    test_plot_dir = os.path.join(exp_dir, "test", "plots")
    os.makedirs(test_plot_dir, exist_ok=True)

    print(f"\nSaving test plots to:\n  {test_plot_dir}\n")

    plot_all(
        results,
        learning_rates,
        optimizers,
        experiment_name=None,        # not needed
        save_combined=True,
        save_separate=True,
        post_training=False          # DO NOT re-save npz
    )

    print("\n✔ Test plots generated successfully.\n")


if __name__ == "__main__":
    main()
