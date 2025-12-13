import torch
from .base import OptimizerState, StatefulOptimizer, SAMOptimizer

# -----------------------------------------------------------------------------
# Adaptive Optimizer Factories (for LinearModel and general PyTorch models)
# -----------------------------------------------------------------------------
# These factories create optimizers that wrap PyTorch's built-in optimizers
# (Adam, AdaGrad, etc.) with proper numerical stability and loss functions.
#
# Usage with expand_sweep_grid:
#   optimizer_factories = {
#       Optimizer.Adam: Adam,
#       Optimizer.SAM_Adam: SAM_Adam,
#   }
#   sweeps = {
#       Optimizer.Adam: {Hyperparam.LearningRate: [1e-3, 1e-2]},
#       Optimizer.SAM_Adam: {
#           Hyperparam.LearningRate: [1e-3],
#           Hyperparam.Rho: [0.05, 0.1]
#       },
#   }
# -----------------------------------------------------------------------------


def Adam(betas=(0.9, 0.999), eps=1e-8) -> StatefulOptimizer:
    """
    Factory for Adam optimizer with exponential loss.

    Args:
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)

    Returns:
        StatefulOptimizer wrapping torch.optim.Adam
    """
    return StatefulOptimizer(torch.optim.Adam, betas=betas, eps=eps)


def AdaGrad(eps=1e-8) -> StatefulOptimizer:
    """
    Factory for AdaGrad optimizer with exponential loss.

    Args:
        eps: Term added to denominator for numerical stability (default: 1e-8)

    Returns:
        StatefulOptimizer wrapping torch.optim.Adagrad
    """
    return StatefulOptimizer(torch.optim.Adagrad, eps=eps)


def SAM_Adam(rho=0.05, betas=(0.9, 0.999), eps=1e-8) -> SAMOptimizer:
    """
    Factory for SAM-Adam optimizer with exponential loss.

    SAM (Sharpness-Aware Minimization) improves generalization by
    seeking parameters in flat minima regions.

    Args:
        rho: Neighborhood size for SAM perturbation (default: 0.05)
        betas: Coefficients for Adam running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)

    Returns:
        SAMOptimizer wrapping torch.optim.Adam
    """
    return SAMOptimizer(torch.optim.Adam, rho=rho, betas=betas, eps=eps)


def SAM_AdaGrad(rho=0.05, eps=1e-8) -> SAMOptimizer:
    """
    Factory for SAM-AdaGrad optimizer with exponential loss.

    Args:
        rho: Neighborhood size for SAM perturbation (default: 0.05)
        eps: Term added to denominator for numerical stability (default: 1e-8)

    Returns:
        SAMOptimizer wrapping torch.optim.Adagrad
    """
    return SAMOptimizer(torch.optim.Adagrad, rho=rho, eps=eps)


