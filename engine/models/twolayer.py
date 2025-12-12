import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Model, ArrayLike
from ..types import ComputeBackend
from typing import List, override

class TwoLayerModel(Model):
    """f(x) = mean(W2 @ relu(W1 @ x)) — PyTorch only"""

    def __init__(
        self,
        D: int,
        k: int,
        output_dim: int = 1,
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
        self.net = torch.nn.Sequential(
            # Layer 1
            torch.nn.Linear(D, k, bias=False),
            # Layer 2
            torch.nn.Linear(k, 1, bias=False),
        ).double().to(device)


    def forward(self, X: ArrayLike) -> ArrayLike:
        """f(X) = W2 @ W1 @ X"""
        Z = self.net(X)
        return Z

    def parameters(self) -> List[ArrayLike]:
        """Return [W1, W2]"""
        return self.net.parameters()

    def zero_grad(self):
        """Clear gradients"""
        self.net.zero_grad()

    @property
    def num_parameters(self) -> int:
        return self.k * self.D + self.output_dim * self.k

    @property
    def effective_weight(self) -> torch.Tensor:
        """
        Computes the effective linear predictor W_eff = u^T V
        Returns shape (D,)
        """
        # Access the internal linear layers
        # Assumes internal structure: self.net[0] is W1, self.net[1] is W2
        W1 = self.net[0].weight # Shape (k, D)
        W2 = self.net[1].weight # Shape (1, k)
        
        # W_eff = W2 @ W1
        with torch.no_grad():
            W_eff = torch.matmul(W2, W1)
        return W_eff.detach().view(-1)
