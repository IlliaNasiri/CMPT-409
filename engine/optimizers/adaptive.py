import torch
from typing import Optional
from .base import OptimizerState, StatefulOptimizer, SAMOptimizer
from ..losses import Loss

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


def Adam(betas=(0.9, 0.999), eps=1e-8, loss: Optional[Loss] = None) -> StatefulOptimizer:
    """
    Factory for Adam optimizer.

    Args:
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        loss: Loss function to use (defaults to ExponentialLoss)

    Returns:
        StatefulOptimizer wrapping torch.optim.Adam
    """
    return StatefulOptimizer(torch.optim.Adam, loss=loss, betas=betas, eps=eps)


def AdaGrad(eps=1e-8, loss: Optional[Loss] = None) -> StatefulOptimizer:
    """
    Factory for AdaGrad optimizer.

    Args:
        eps: Term added to denominator for numerical stability (default: 1e-8)
        loss: Loss function to use (defaults to ExponentialLoss)

    Returns:
        StatefulOptimizer wrapping torch.optim.Adagrad
    """
    return StatefulOptimizer(torch.optim.Adagrad, loss=loss, eps=eps)


def SAM_Adam(rho=0.05, betas=(0.9, 0.999), eps=1e-8, loss: Optional[Loss] = None) -> SAMOptimizer:
    """
    Factory for SAM-Adam optimizer.

    SAM (Sharpness-Aware Minimization) improves generalization by
    seeking parameters in flat minima regions.

    Args:
        rho: Neighborhood size for SAM perturbation (default: 0.05)
        betas: Coefficients for Adam running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        loss: Loss function to use (defaults to ExponentialLoss)

    Returns:
        SAMOptimizer wrapping torch.optim.Adam
    """
    return SAMOptimizer(torch.optim.Adam, loss=loss, rho=rho, betas=betas, eps=eps)


def SAM_AdaGrad(rho=0.05, eps=1e-8, loss: Optional[Loss] = None) -> SAMOptimizer:
    """
    Factory for SAM-AdaGrad optimizer.

    Args:
        rho: Neighborhood size for SAM perturbation (default: 0.05)
        eps: Term added to denominator for numerical stability (default: 1e-8)
        loss: Loss function to use (defaults to ExponentialLoss)

    Returns:
        SAMOptimizer wrapping torch.optim.Adagrad
    """
    return SAMOptimizer(torch.optim.Adagrad, loss=loss, rho=rho, eps=eps)


