# ===============================================================
# run_ngd_experiments.py   (GD / NGD / SAM / SAM-NGD)
# ===============================================================
from engine import (
    run_training,
    TwoLayerModel,
    DatasetSplit,
    Metric,
    Optimizer,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    exponential_loss,
    get_error_rate,
)
from engine.optimizers import step_gd, step_ngd_stable, step_sam_stable, step_sam_ngd_stable
from engine.optimizers.base import make_optimizer
from engine.plotting import plot_all
import torch
import os

# Configure PyTorch to use all CPU cores
torch.set_num_threads(os.cpu_count())

def main():
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------
    N, D = 200, 5000
    X, y, v_pop = make_soudry_dataset(N, D, device=device)
    print("Dataset ready:", X.shape, y.shape)

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # ----------------------------------------------------------
    # 2-layer width
    # ----------------------------------------------------------
    k = 50

    # Model factory (Torch only now)
    def model_factory():
        return TwoLayerModel(D, k, device=device)

    # Metrics factory (NO w_star for two-layer)
    def metrics_factory(model):
        return MetricsCollector(
            metric_fns={
                Metric.Loss: exponential_loss,
                Metric.Error: get_error_rate,
            },
            w_star=None  # No reference solution for multi-layer
        )

    # ----------------------------------------------------------
    # Learning rate sweep
    # ----------------------------------------------------------
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    total_iters = 10_000

    # ----------------------------------------------------------
    # Register optimizers with enum keys
    # ----------------------------------------------------------
    optimizers = {
        Optimizer.GD: make_optimizer(step_gd),
        Optimizer.NGD: make_optimizer(step_ngd_stable),
        Optimizer.SAM: make_optimizer(step_sam_stable),
        Optimizer.SAM_NGD: make_optimizer(step_sam_ngd_stable),
    }

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=total_iters,
        debug=True
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="2layers_gd_family"
    )


if __name__ == "__main__":
    main()
