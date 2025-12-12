from abc import ABC, abstractmethod
from typing import List, Union
import torch
from ..types import ArrayLike

class Model(ABC):
    """Abstract base for optimization models (PyTorch only)"""

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def forward(self, X: ArrayLike) -> ArrayLike:
        """Computes predictions."""
        pass

    @abstractmethod
    def parameters(self) -> List[ArrayLike]:
        """Returns a list of mutable parameter tensors."""
        pass

    @abstractmethod
    def zero_grad(self):
        """Clears gradients."""
        pass

    def train(self, mode: bool = True):
        """Optional: Toggle training mode (for Dropout/BatchNorm)."""
        pass
