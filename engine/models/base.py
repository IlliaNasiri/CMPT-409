from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import torch
from ..types import ComputeBackend, ArrayLike

class Model(ABC):
    """Abstract base for optimization models"""

    def __init__(self, backend: ComputeBackend, device: str = "cpu"):
        self.backend = backend
        self.device = device

    @abstractmethod
    def forward(self, X: ArrayLike) -> ArrayLike:
        """Computes predictions."""
        pass

    @abstractmethod
    def parameters(self) -> List[ArrayLike]:
        """Returns a list of mutable parameter tensors/arrays."""
        pass

    @abstractmethod
    def zero_grad(self):
        """Clears gradients."""
        pass

    def train(self, mode: bool = True):
        """Optional: Toggle training mode (for Dropout/BatchNorm)."""
        pass
