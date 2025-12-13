import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Mapping
from datetime import datetime
from .types import OptimizerConfig, MetricKey, Metric, Hyperparam
from .history import TrainingHistory
from .strategies import PlotStrategy, PlotContext

# Type alias for results - use Mapping for covariance
ResultsType = Mapping[OptimizerConfig, Union[TrainingHistory, List[TrainingHistory]]]


def plot_all(
    results: ResultsType,
    experiment_name: str,
    save_combined: bool = True,
    save_separate: bool = True,
    save_aggregated: bool = True,
    post_training: bool = False,
    strategy_overrides: Optional[Dict[Metric, PlotStrategy]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Unified plotting for all model types.
    Automatically detects which metrics are available from TrainingHistory.

    Args:
        results: Training results dict[OptimizerConfig] = TrainingHistory
        experiment_name: Name for experiment directory
        save_combined: Save combined plots (grouped by learning rate)
        save_separate: Save separate plots per optimizer
        save_aggregated: Save aggregated comparison plots
        post_training: If True, skip saving results.npz (already saved)
        strategy_overrides: Optional dict mapping Metric -> PlotStrategy to override defaults
        output_dir: Optional custom output directory (overrides experiment_name-based path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        base_dir = output_dir
    elif experiment_name:
        base_dir = Path("experiments") / experiment_name / timestamp
    else:
        base_dir = Path("experiments") / "test_plots" / timestamp

    base_dir.mkdir(parents=True, exist_ok=True)

    # Group configs by learning rate
    configs_by_lr: Dict[float, List[OptimizerConfig]] = {}
    for config in results.keys():
        lr = config.learning_rate
        if lr not in configs_by_lr:
            configs_by_lr[lr] = []
        configs_by_lr[lr].append(config)

    learning_rates = sorted(configs_by_lr.keys())
    all_configs = list(results.keys())

    # Save results as NPZ if not already saved
    if not post_training:
        save_results_npz(results, base_dir / "results.npz")

    # Detect metric keys from first history
    first_config = all_configs[0]
    first_entry = results[first_config]
    if isinstance(first_entry, list):
        if len(first_entry) == 0:
            print("Warning: No history found to detect metrics.")
            return
        first_hist = first_entry[0]
    else:
        first_hist = first_entry
    metric_keys = first_hist.metric_keys

    # Group metrics by type
    metric_types: Dict[str, List[MetricKey]] = {}
    for key in metric_keys:
        metric_name = str(key.metric.name.lower())
        if metric_name not in metric_types:
            metric_types[metric_name] = []
        metric_types[metric_name].append(key)

    # Resolve strategy overrides
    overrides = strategy_overrides or {}

    # Generate Plots
    for metric_name, keys in metric_types.items():
        metric = keys[0].metric
        strategy = overrides.get(metric, metric.strategy)

        if save_combined:
            plot_combined(
                results,
                configs_by_lr,
                learning_rates,
                keys,
                strategy,
                base_dir / "combined" / f"{metric_name}.png",
            )

        if save_separate:
            for config in all_configs:
                plot_separate(
                    results,
                    config,
                    keys,
                    strategy,
                    base_dir / "separate" / config.name / f"{metric_name}.png",
                )

        if save_aggregated:
            plot_aggregated(
                results,
                configs_by_lr,
                learning_rates,
                keys,
                strategy,
                base_dir / "aggregated" / f"{metric_name}_comparison.png",
            )


def _get_history(
    entry: Union[TrainingHistory, List[TrainingHistory]],
) -> Optional[TrainingHistory]:
    """Extract single history from entry, handling both single and list cases."""
    if isinstance(entry, list):
        return entry[0] if entry else None
    return entry


def _get_histories(
    entry: Union[TrainingHistory, List[TrainingHistory]],
) -> List[TrainingHistory]:
    """Extract list of histories from entry."""
    if isinstance(entry, list):
        return entry
    return [entry]


def plot_aggregated(
    results: ResultsType,
    configs_by_lr: Mapping[float, List[OptimizerConfig]],
    learning_rates: List[float],
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Paper-style plotting: one subplot per optimizer type, colored by learning rate."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Group configs by optimizer type (ignoring hyperparams)
    optimizer_types: Dict[str, List[OptimizerConfig]] = {}
    for config in results.keys():
        opt_name = config.optimizer.name
        if opt_name not in optimizer_types:
            optimizer_types[opt_name] = []
        optimizer_types[opt_name].append(config)

    opt_names = sorted(optimizer_types.keys())
    ncols = len(opt_names)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), sharey=True)
    if ncols == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    lr_colors = {lr: cmap(i % 10) for i, lr in enumerate(learning_rates)}

    for i, opt_name in enumerate(opt_names):
        ax = axes[i]
        key = metric_keys[0]
        strategy.configure_axis(ax, base_label=key.metric.name)

        # Get all configs for this optimizer type
        opt_configs = optimizer_types[opt_name]

        for lr in learning_rates:
            # Find configs with this lr
            lr_configs = [c for c in opt_configs if c.learning_rate == lr]
            if not lr_configs:
                continue

            color = lr_colors[lr]

            for config in lr_configs:
                entry = results[config]
                histories = _get_histories(entry)

                if not histories:
                    continue

                all_values: List[np.ndarray] = []
                steps: Optional[np.ndarray] = None

                for h in histories:
                    h_cpu = h.copy_cpu()
                    vals = np.array(h_cpu.get(key))
                    curr_steps = np.array(h_cpu.get_steps())
                    all_values.append(vals)
                    if steps is None or len(curr_steps) > len(steps):
                        steps = curr_steps

                # Plot individual runs (faint) if multiple
                if len(histories) > 1 and steps is not None:
                    for vals in all_values:
                        run_steps = steps[: len(vals)]
                        ctx = PlotContext(
                            ax=ax,
                            x=run_steps,
                            y=vals,
                            label="",
                            color=color,
                            alpha=0.15,
                            linestyle="-",
                        )
                        strategy.plot(ctx)

                # Plot mean (or single run)
                if all_values and steps is not None:
                    min_len = min(len(v) for v in all_values)
                    truncated = np.stack([v[:min_len] for v in all_values])
                    mean_vals = np.mean(truncated, axis=0)
                    mean_steps = steps[:min_len]

                    # Label with config name for non-trivial hyperparams
                    label = config.name if len(lr_configs) > 1 else f"lr={lr}"
                    ctx = PlotContext(
                        ax=ax,
                        x=mean_steps,
                        y=mean_vals,
                        label=label,
                        color=color,
                        alpha=1.0,
                        linestyle="-",
                    )
                    strategy.plot(ctx)

        ax.set_title(opt_name)
        ax.set_xlabel("Steps")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_combined(
    results: ResultsType,
    configs_by_lr: Mapping[float, List[OptimizerConfig]],
    learning_rates: List[float],
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Combined view: one subplot per learning rate, all optimizers overlaid."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    ncols = len(learning_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, lr in enumerate(learning_rates):
        ax = axes[i]
        strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

        configs = configs_by_lr.get(lr, [])
        for config in configs:
            entry = results[config]
            history = _get_history(entry)

            if history is None:
                continue

            history_cpu = history.copy_cpu()
            steps = np.array(history_cpu.get_steps())

            for key in metric_keys:
                values = np.array(history_cpu.get(key))
                label = config.name + (f"_{key.split.name}" if key.split else "")
                ctx = PlotContext(ax=ax, x=steps, y=values, label=label, alpha=0.7)
                strategy.plot(ctx)

        ax.set_xlabel("Steps")
        ax.set_title(f"LR = {lr}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_separate(
    results: ResultsType,
    config: OptimizerConfig,
    metric_keys: List[MetricKey],
    strategy: PlotStrategy,
    filepath: Path,
) -> None:
    """Single optimizer config view."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    strategy.configure_axis(ax, base_label=metric_keys[0].metric.name)

    entry = results[config]
    histories = _get_histories(entry)

    if not histories:
        plt.close()
        return

    for h in histories:
        h_cpu = h.copy_cpu()
        steps = np.array(h_cpu.get_steps())

        for key in metric_keys:
            values = np.array(h_cpu.get(key))
            label = key.split.name if key.split else key.metric.name
            ctx = PlotContext(ax=ax, x=steps, y=values, label=label, alpha=0.7)
            strategy.plot(ctx)

    ax.set_xlabel("Steps")
    ax.set_title(config.name)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def save_results_npz(
    results: ResultsType,
    filepath: Path,
) -> None:
    """Save results as NPZ with config names as keys."""
    import torch

    data: Dict[str, np.ndarray] = {}
    for config, entry in results.items():
        histories = _get_histories(entry)

        for seed, history in enumerate(histories):
            hist_cpu = history.copy_cpu()
            hist_dict = hist_cpu.to_dict()

            for key, values in hist_dict.items():
                # Use config name which includes all hyperparams
                npz_key = f"{config.name}_seed{seed}_{key}"
                # Convert Tensor to numpy if needed
                if isinstance(values, torch.Tensor):
                    values = values.cpu().numpy()
                data[npz_key] = values

    np.savez(filepath, **data)  # type: ignore[call-arg]
