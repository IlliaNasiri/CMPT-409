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
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Forward pass: X @ params -> predictions"""
        pass

    @abstractmethod
    def parameters(self) -> List[ArrayLike]:
        """Get list of all parameters"""
        pass

    @abstractmethod
    def zero_grad(self):
        """Clear gradients (for PyTorch models)"""
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Total parameter count"""
        pass
