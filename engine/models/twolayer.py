import torch
import torch.nn as nn
from .base import Model, ArrayLike
from typing import List, cast


class TwoLayerModel(Model):
    """f(x) = x @ W1^T @ W2^T — PyTorch only"""

    net: nn.Sequential  # Type annotation for the sequential network

    def __init__(self, D: int, k: int, output_dim: int = 1, device: str = "cpu"):
        super().__init__(device)
        self.D = D
        self.k = k
        self.output_dim = output_dim

        # Initialize as nn.Sequential for autograd
        self.net = (
            torch.nn.Sequential(
                # Layer 1
                torch.nn.Linear(D, k, bias=False),
                # Layer 2
                torch.nn.Linear(k, 1, bias=False),
            )
            .double()
            .to(device)
        )

    def forward(self, X: ArrayLike) -> ArrayLike:
        """f(x) = x @ W1^T @ W2^T — PyTorch only"""
        Z = self.net(X)
        return Z.squeeze(-1)  # Return shape (N,) instead of (N, 1)

    def parameters(self) -> List[ArrayLike]:
        """Return [W1, W2]"""
        return list(self.net.parameters())

    def zero_grad(self):
        """Clear gradients"""
        self.net.zero_grad()

    @property
    def num_parameters(self) -> int:
        return self.k * self.D + self.output_dim * self.k

    @property
    def W1(self) -> torch.Tensor:
        """Returns the first layer weight tensor (D, k)."""
        return cast(nn.Linear, self.net[0]).weight

    @property
    def W2(self) -> torch.Tensor:
        """Returns the second layer weight tensor (k, 1)."""
        return cast(nn.Linear, self.net[1]).weight

    @property
    def effective_weight(self) -> torch.Tensor:
        """
        Computes the effective linear predictor W_eff = W2 @ W1
        Returns shape (D,)
        """
        # W_eff = W2 @ W1
        with torch.no_grad():
            W_eff = torch.matmul(self.W2, self.W1)
            return W_eff.detach()
