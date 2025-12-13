from engine import (
    run_training,
    TwoLayerModel,
    DatasetSplit,
    Metric,
    Optimizer,
    OptimizerConfig,
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
from engine.optimizers.manual import (
    ManualAdam,
    ManualAdaGrad,
    ManualSAM_Adam,
    ManualSAM_AdaGrad
)
from engine.plotting import plot_all
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate dataset
    X, y, v_pop = make_soudry_dataset(n=200, d=5000, device=device)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # Model factory
    # Note: Ensure TwoLayerModel is purely linear (Sequential(Linear, Linear)) 
    # to match the manual optimizer's gradient derivation.
    def model_factory():
        return TwoLayerModel(X.shape[1], 100, device=device)

    # Metrics factory
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
                Metric.Angle: get_angle,
                Metric.Distance: get_direction_distance,
            },
            w_star=w_star
        )

    # Optimizer factories
    # Using the fused manual implementations for ~2x speedup on small models
    optimizer_factories = {
        Optimizer.Adam: ManualAdam,
        Optimizer.AdaGrad: ManualAdaGrad,
        Optimizer.SAM_Adam: ManualSAM_Adam,
        Optimizer.SAM_AdaGrad: ManualSAM_AdaGrad,
    }

    # Hyperparameter sweeps
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    rho_values = [0.05]

    sweeps = {
        Optimizer.Adam: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
        },
        Optimizer.SAM_Adam: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
        },
        Optimizer.SAM_AdaGrad: {
            Hyperparam.LearningRate: learning_rates,
            Hyperparam.Rho: rho_values,
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
        debug=True
    )

    # Plotting
    plot_all(
        results,
        experiment_name="2layers_adam_family_testing"
    )

if __name__ == "__main__":
    main()

