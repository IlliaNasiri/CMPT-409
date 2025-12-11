import numpy as np
from typing import Dict, List, Callable, Tuple
from tqdm import tqdm
from .types import DatasetSplit, Optimizer, MetricKey, ComputeBackend
from .models.base import Model
from .metrics import MetricsCollector
from .history import TrainingHistory

def run_training(
    datasets: Dict[DatasetSplit, Tuple[np.ndarray, np.ndarray]],
    model_factory: Callable[[], Model],
    optimizers: Dict[Optimizer, Callable],
    learning_rates: List[float],
    metrics_collector_factory: Callable[[Model], MetricsCollector],
    train_split: DatasetSplit = DatasetSplit.Train,
    total_iters: int = 100_000,
    debug: bool = True
) -> Dict[float, Dict[Optimizer, TrainingHistory]]:
    """
    Unified training loop supporting both linear and multi-layer models.

    Args:
        datasets: Dict mapping splits to (X, y) tuples
        model_factory: Function returning fresh model instance
        optimizers: Dict mapping Optimizer enum to step functions
        learning_rates: List of learning rates to sweep
        metrics_collector_factory: Function creating MetricsCollector from model
        train_split: Which split to use for training
        total_iters: Number of training iterations
        debug: Show progress bars

    Returns:
        results[lr][opt] = TrainingHistory
    """
    # Logarithmic recording steps (200 points, always includes t=1)
    record_steps = set(np.unique(np.logspace(0, np.log10(total_iters), 200).astype(int)))
    record_steps.add(1)
    record_steps = sorted(record_steps)

    X_train, y_train = datasets[train_split]
    results = {}

    for lr in learning_rates:
        results[lr] = {}

        for opt_enum, step_fn in optimizers.items():
            if debug:
                print(f"\nRunning {opt_enum.name} with lr={lr}")

            # Create fresh model and metrics collector
            model = model_factory()
            collector = metrics_collector_factory(model)

            # Pre-allocate history
            metric_keys = collector.get_metric_keys(list(datasets.keys()))
            history = TrainingHistory(
                metric_keys=metric_keys,
                num_records=len(record_steps),
                backend=model.backend,
                device=model.device
            )

            # Training loop
            iterator = tqdm(range(1, total_iters + 1)) if debug else range(1, total_iters + 1)

            for t in iterator:
                try:
                    # Optimizer step
                    w = model.parameters()[0]  # Get current weights
                    w_new = step_fn(w, X_train, y_train, lr)

                    # Update model
                    if model.backend == ComputeBackend.NumPy:
                        model.w = w_new
                    else:
                        model.w.data = w_new

                    # Record metrics at logging steps
                    if t in record_steps:
                        metrics = collector.compute_all(w_new, datasets)
                        history.record(t, metrics)

                except Exception as e:
                    if debug:
                        print(f"Error at step {t}: {e}")
                    break

            results[lr][opt_enum] = history

    return results
