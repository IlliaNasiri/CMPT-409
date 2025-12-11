import numpy as np
import torch
from .base import Model, ArrayLike
from ..types import ComputeBackend

class LinearModel(Model):
    """f(x) = w^T x — supports both NumPy and PyTorch"""

    def __init__(self, D: int, backend: ComputeBackend = ComputeBackend.NumPy, device: str = "cpu"):
        super().__init__(backend, device)
        self.D = D

        # Initialize weights
        if backend == ComputeBackend.Torch:
            self.w = torch.randn(D, dtype=torch.float64, device=device) * 1e-6
            self.w.requires_grad = True
        else:
            self.w = np.random.randn(D) * 1e-6

    def predict(self, X: ArrayLike) -> ArrayLike:
        """f(X) = X @ w"""
        return X @ self.w

    def parameters(self) -> list[ArrayLike]:
        """Return [w]"""
        return [self.w]

    def zero_grad(self):
        """Clear gradients (PyTorch only)"""
        if self.backend == ComputeBackend.Torch and self.w.grad is not None:
            self.w.grad.zero_()

    @property
    def num_parameters(self) -> int:
        return self.D
