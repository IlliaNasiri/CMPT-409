import torch
import torch.nn as nn
from .base import Model, ArrayLike
from ..types import ComputeBackend
from typing import List

class TwoLayerModel(Model):
    """f(x) = mean(W2 @ relu(W1 @ x)) — PyTorch only"""

    def __init__(
        self,
        D: int,
        k: int,
        output_dim: int = 10,
        backend: ComputeBackend = ComputeBackend.Torch,
        device: str = "cpu"
    ):
        if backend != ComputeBackend.Torch:
            raise ValueError("TwoLayerModel requires PyTorch backend")

        super().__init__(backend, device)
        self.D = D
        self.k = k
        self.output_dim = output_dim

        # Initialize as nn.Module for autograd
        self.W1 = nn.Parameter(torch.randn(k, D, dtype=torch.float64, device=device) * 0.01)
        self.W2 = nn.Parameter(torch.randn(output_dim, k, dtype=torch.float64, device=device) * 0.01)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """f(X) = mean(W2 @ relu(W1 @ X^T), dim=0)"""
        Z = torch.relu(self.W1 @ X.T)  # (k, N)
        pred = self.W2 @ Z              # (output_dim, N)
        return pred.mean(dim=0)         # (N,) - mean over output_dim

    def parameters(self) -> List[torch.Tensor]:
        """Return [W1, W2]"""
        return [self.W1, self.W2]

    def zero_grad(self):
        """Clear gradients"""
        if self.W1.grad is not None:
            self.W1.grad.zero_()
        if self.W2.grad is not None:
            self.W2.grad.zero_()

    @property
    def num_parameters(self) -> int:
        return self.k * self.D + self.output_dim * self.k
