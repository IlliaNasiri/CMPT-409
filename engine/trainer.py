import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Callable, Tuple
from tqdm import tqdm

from engine.optimizers.base import StatefulOptimizer, StatelessOptimizer
from .types import DatasetSplit, Optimizer, MetricKey, ComputeBackend, ArrayLike
from .optimizers import OptimizerState
from .models.base import Model
from .metrics import MetricsCollector
from .history import TrainingHistory

def run_training(
    datasets: Dict[DatasetSplit, Tuple[ArrayLike, ArrayLike]],
    model_factory: Callable[[], Model],
    optimizers: Dict[Optimizer, OptimizerState],
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

    torch_datasets = {}
    numpy_datasets = {}

    results = {}

    for lr in learning_rates:
        results[lr] = {}

        for opt_enum, generic_optim in optimizers.items():
            if debug:
                print(f"\nRunning {opt_enum.name} with lr={lr}")

            # Create fresh model and metrics collector
            model = model_factory()

            # Convert inputs to Tensor if necessary
            match model.backend:
                case ComputeBackend.NumPy:
                    if not numpy_datasets:
                        for split, (X, y) in datasets.items():
                            numpy_datasets[split] = (X, y)
                    curr_datasets = numpy_datasets
                case ComputeBackend.Torch:
                    if not torch_datasets:
                        for split, (X, y) in datasets.items():
                            torch_datasets[split] = (torch.from_numpy(X).to(model.device), torch.from_numpy(y).to(model.device))
                    curr_datasets = torch_datasets

            X_train, y_train = curr_datasets[train_split]
            collector = metrics_collector_factory(model)
            #optim = generic_optim(lr)
            generic_optim.reset()

            # Pre-allocate history buffer
            metric_keys = collector.get_metric_keys(list(curr_datasets.keys()))
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
                    w_new = generic_optim.step(model, X_train, y_train, lr)

                    # Record metrics at logging steps
                    if t in record_steps:
                        metrics = collector.compute_all(model, curr_datasets)
                        history.record(t, metrics)

                except Exception as e:
                    print(f"Error at step {t}: {e}")
                    raise e

            results[lr][opt_enum] = history

    return results



class ExponentialLoss(nn.Module):
    """
    Computes the mean exponential loss: mean(exp(-y * y_pred))
    """
    def __init__(self, clamp_min=-50, clamp_max=100):
        super().__init__()
        # Clamping is essential because exp() grows excessively fast.
        # exp(100) is ~2e43, which is safe in float64 but huge.
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: The raw scores (logits) from the network. Shape (N, 1) or (N,)
            target: The targets, MUST be {-1, 1}. Shape (N, 1) or (N,)
        """
        # Ensure shapes match
        if input.shape != target.shape:
            target = target.view_as(input)

        # Compute margins: y * f(x)
        margins = target * input

        # Numerical stability (prevents Inf/NaN gradients)
        margins = torch.clamp(margins, min=self.clamp_min, max=self.clamp_max)

        # Loss
        return torch.mean(torch.exp(-margins))

