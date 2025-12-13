"""Hyperparameter sweep utilities."""

from itertools import product
from typing import Callable, Dict, List, Any, Mapping
from .types import Optimizer, OptimizerConfig, Hyperparam
from .optimizers import OptimizerState

__all__ = ["expand_sweep_grid"]


def expand_sweep_grid(
    optimizer_factories: Mapping[Optimizer, Callable[..., OptimizerState]],
    sweeps: Mapping[Optimizer, Mapping[Hyperparam, List[float]]],
) -> Dict[OptimizerConfig, Callable[[], OptimizerState]]:
    """
    Expand optimizers with hyperparameter sweeps into concrete configurations.

    Args:
        optimizer_factories: Map from Optimizer enum to factory callable.
            Factory signature: (**hyperparams) -> OptimizerState
            where hyperparams are string keys (e.g., rho=0.1)
        sweeps: Hyperparameter sweeps per optimizer using Hyperparam enum keys.
            Must include Hyperparam.LearningRate for each optimizer.
            Example: {
                Optimizer.GD: {
                    Hyperparam.LearningRate: [1e-3, 1e-2],
                },
                Optimizer.SAM: {
                    Hyperparam.LearningRate: [1e-3, 1e-2],
                    Hyperparam.Rho: [0.05, 0.1],
                },
            }

    Returns:
        Dict mapping OptimizerConfig to zero-argument factory callables.
        The OptimizerConfig contains all hyperparameters (including lr).
        The factory produces an optimizer with non-lr hyperparameters bound.

    Example:
        >>> factories = {
        ...     Optimizer.GD: make_optimizer_factory(step_gd),
        ...     Optimizer.SAM: make_optimizer_factory(step_sam_stable),
        ... }
        >>> sweeps = {
        ...     Optimizer.GD: {Hyperparam.LearningRate: [0.01, 0.1]},
        ...     Optimizer.SAM: {Hyperparam.LearningRate: [0.01], Hyperparam.Rho: [0.1, 0.5]},
        ... }
        >>> configs = expand_sweep_grid(factories, sweeps)
        >>> # Returns configs for: GD(lr=0.01), GD(lr=0.1), SAM(lr=0.01,rho=0.1), SAM(lr=0.01,rho=0.5)
    """
    result: Dict[OptimizerConfig, Callable[[], OptimizerState]] = {}

    for opt_enum, factory in optimizer_factories.items():
        if opt_enum not in sweeps:
            raise ValueError(f"Optimizer {opt_enum.name} missing from sweeps dict")

        opt_sweeps = sweeps[opt_enum]
        if Hyperparam.LearningRate not in opt_sweeps:
            raise ValueError(
                f"Optimizer {opt_enum.name} missing Hyperparam.LearningRate in sweep"
            )

        # Expand hyperparameter grid
        param_enums = list(opt_sweeps.keys())
        param_values = list(opt_sweeps.values())

        for values in product(*param_values):
            # Build hyperparams with Hyperparam enum keys
            hyperparams_enum = dict(zip(param_enums, values))

            # Create config with all hyperparams (for storage/display)
            config = OptimizerConfig(
                optimizer=opt_enum, hyperparams=tuple(hyperparams_enum.items())
            )

            # Create factory - extract non-LR params as string keys for step fn
            step_fn_kwargs = {
                hp.value: v
                for hp, v in hyperparams_enum.items()
                if hp != Hyperparam.LearningRate
            }

            def make_bound_factory(
                f: Callable = factory, kwargs: Dict[str, Any] = step_fn_kwargs
            ) -> Callable[[], OptimizerState]:
                return lambda: f(**kwargs)

            result[config] = make_bound_factory()

    return result
