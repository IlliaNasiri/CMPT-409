from engine import (
    run_training,
    LinearModel,
    DatasetSplit,
    Metric,
    Optimizer,
    ComputeBackend,
    MetricsCollector,
    split_train_test,
    make_soudry_dataset,
    get_empirical_max_margin,
    exponential_loss,
    get_error_rate,
    get_angle,
    get_direction_distance,
)
from engine.optimizers import step_gd, step_sam_stable, step_ngd_stable, step_sam_ngd_stable
from engine.plotting import plot_all

def main():
    # Generate dataset
    X, y, v_pop = make_soudry_dataset(n=200, d=5000)
    w_star = get_empirical_max_margin(X, y)

    # Split data
    datasets = split_train_test(X, y, test_size=0.2, random_state=42)

    # Model factory
    def model_factory():
        return LinearModel(X.shape[1], backend=ComputeBackend.NumPy)

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

    # Optimizers
    optimizers = {
        Optimizer.GD: step_gd,
        Optimizer.SAM: step_sam_stable,
        Optimizer.NGD: step_ngd_stable,
        Optimizer.SAM_NGD: step_sam_ngd_stable,
    }

    # Run training
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    results = run_training(
        datasets=datasets,
        model_factory=model_factory,
        optimizers=optimizers,
        learning_rates=learning_rates,
        metrics_collector_factory=metrics_factory,
        train_split=DatasetSplit.Train,
        total_iters=100_000,
        debug=True
    )

    # Plotting
    plot_all(
        results,
        learning_rates,
        list(optimizers.keys()),
        experiment_name="soudry"
    )

if __name__ == "__main__":
    main()
