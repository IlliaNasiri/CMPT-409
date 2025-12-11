import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from .types import Optimizer, MetricKey
from .history import TrainingHistory

def plot_all(
    results: Dict[float, Dict[Optimizer, TrainingHistory]],
    learning_rates: List[float],
    optimizers: List[Optimizer],
    experiment_name: str,
    save_combined: bool = True,
    save_separate: bool = True,
):
    """
    Unified plotting for all model types.
    Automatically detects which metrics are available from TrainingHistory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("experiments") / experiment_name / timestamp
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save results as NPZ (flattened)
    save_results_npz(results, learning_rates, optimizers, base_dir / "results.npz")

    # Get available metrics from first history
    first_history = results[learning_rates[0]][optimizers[0]]
    metric_keys = first_history.metric_keys

    # Group metrics by type (ignoring splits for plotting)
    metric_types = {}
    for key in metric_keys:
        metric_name = str(key.metric.name.lower())
        if metric_name not in metric_types:
            metric_types[metric_name] = []
        metric_types[metric_name].append(key)

    # Plot each metric type
    for metric_name, keys in metric_types.items():
        if save_combined:
            plot_combined(results, learning_rates, optimizers, keys,
                          base_dir / "combined" / f"{metric_name}.png")
        if save_separate:
            for opt in optimizers:
                plot_separate(results, learning_rates, opt, keys,
                              base_dir / "separate" / opt.name / f"{metric_name}.png")

def save_results_npz(results, learning_rates, optimizers, filepath):
    """Save flattened results as NPZ"""
    data = {}
    for lr in learning_rates:
        for opt in optimizers:
            history = results[lr][opt]
            hist_dict = history.to_dict()
            for key, values in hist_dict.items():
                npz_key = f"lr{lr}_{opt.name}_{key}"
                data[npz_key] = values
    np.savez(filepath, **data)

def plot_combined(results, learning_rates, optimizers, metric_keys, filepath):
    """Plot all optimizers and LRs for given metric keys"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ncols = len(learning_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        for opt in optimizers:
            history = results[lr][opt]
            steps = history.get_steps()

            # Plot all keys for this metric (e.g., loss_train, loss_val, loss_test)
            for key in metric_keys:
                values = history.get(key)
                label = f"{opt.name}_{key.split.name if key.split else ''}"
                ax.loglog(steps, values, label=label, alpha=0.7)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_keys[0].metric.name)
        ax.set_title(f"LR = {lr}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def plot_separate(results, learning_rates, optimizer, metric_keys, filepath):
    """Plot single optimizer across all LRs"""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for lr in learning_rates:
        history = results[lr][optimizer]
        steps = history.get_steps()

        for key in metric_keys:
            values = history.get(key)
            label = f"lr={lr}_{key.split.name if key.split else ''}"
            ax.loglog(steps, values, label=label, alpha=0.7)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(metric_keys[0].metric.name)
    ax.set_title(f"{optimizer.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
