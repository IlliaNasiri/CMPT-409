import torch
from .base import Model, ArrayLike

class LinearModel(Model):
    """f(x) = w^T x — PyTorch only"""

    def __init__(self, D: int, device: str = "cpu"):
        super().__init__(device)
        self.D = D

        # Initialize weights (Torch only)
        self.w = torch.randn(D, dtype=torch.float64, device=device) * 1e-6
        self.w.requires_grad = True

    def forward(self, X: ArrayLike) -> ArrayLike:
        """f(X) = X @ w"""
        return torch.matmul(X, self.w)

    def parameters(self) -> list[ArrayLike]:
        """Return [w]"""
        return [self.w]

    def zero_grad(self):
        """Clear gradients"""
        if self.w.grad is not None:
            self.w.grad.zero_()

    @property
    def num_parameters(self) -> int:
        return self.D
