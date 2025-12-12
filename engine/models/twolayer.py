import torch
import torch.nn as nn
from .base import Model, ArrayLike
from typing import List, override

class TwoLayerModel(Model):
    """f(x) = W2 @ relu(W1 @ x) — PyTorch only"""

    def __init__(self, D: int, k: int, output_dim: int = 1, device: str = "cpu"):
        super().__init__(device)
        self.D = D
        self.k = k
        self.output_dim = output_dim

        # Initialize as nn.Sequential for autograd
        self.net = torch.nn.Sequential(
            # Layer 1
            torch.nn.Linear(D, k, bias=False),
            # Layer 2
            torch.nn.Linear(k, 1, bias=False),
        ).double().to(device)

    def forward(self, X: ArrayLike) -> ArrayLike:
        """f(X) = W2 @ W1 @ X"""
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
    def effective_weight(self) -> torch.Tensor:
        """
        Computes the effective linear predictor W_eff = W2 @ W1
        Returns shape (D,)
        """
        W1 = self.net[0].weight  # Shape (k, D)
        W2 = self.net[1].weight  # Shape (1, k)

        # W_eff = W2 @ W1
        with torch.no_grad():
            W_eff = torch.matmul(W2, W1)
        return W_eff.detach().view(-1)
