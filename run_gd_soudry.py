"""Full-batch GD experiments on Soudry dataset."""

import torch
import os

from engine import (
    run_training,
    LinearModel,
    DatasetSplit,
    Metric,
    Optimizer,
    Hyperparam,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_empirical_max_margin,
    exponential_loss,
    get_error_rate,
    get_angle,
    get_direction_distance,
    expand_sweep_grid,
)
from engine.optimizers import (
    step_gd,
    step_sam_stable,
    step_ngd_stable,
    step_sam_ngd_stable,
    make_optimizer_factory,
)
from engine.plotting import plot_all

# Configure PyTorch to use half the CPU cores
cpu_count = os.cpu_count()
if cpu_count is not None:
    torch.set_num_threads(cpu_count // 2)


def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, margin=1.0, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=40, random_state=42)

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], device=device)

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
            },
            w_star=w_star,
        )

    # Optimizer factories
    optimizer_factories = {
        Optimizer.GD: make_optimizer_factory(step_gd),
        Optimizer.SAM: make_optimizer_factory(step_sam_stable),
        Optimizer.NGD: make_optimizer_factory(step_ngd_stable),
        Optimizer.SAM_NGD: make_optimizer_factory(step_sam_ngd_stable),
    }

    # Hyperparameter sweeps
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    sweeps = {
        Optimizer.GD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],  # Default rho
        },
        Optimizer.NGD: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM_NGD: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: [0.05],  # Default rho
        },
    }

    # Expand to concrete configurations
    optimizer_configs = expand_sweep_grid(optimizer_factories, sweeps)

    # Run training
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizer_configs,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=100_000,
        debug=True,
    )

    # Plotting
    plot_all(
        results,
        experiment_name="soudry",
    )


if __name__ == "__main__":
    main()
