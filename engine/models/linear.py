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
        match backend:
            case ComputeBackend.Torch:
                self.w = torch.randn(D, dtype=torch.float64, device=device) * 1e-6
                self.w.requires_grad = True
            case ComputeBackend.NumPy:
                self.w = np.random.randn(D) * 1e-6

    def forward(self, X: ArrayLike) -> ArrayLike:
        """f(X) = X @ w"""
        match self.backend:
            case ComputeBackend.Torch:
                z = torch.matmul(X, self.w)
            case ComputeBackend.NumPy:
                z = X @ self.w
        return z

    def parameters(self) -> list[ArrayLike]:
        """Return [w]"""
        return [self.w]

    def zero_grad(self):
        """Clear gradients (PyTorch only)"""
        match self.backend:
            case ComputeBackend.Torch:
                if self.w.requires_grad: self.w.grad.zero_()

    @property
    def num_parameters(self) -> int:
        return self.D
