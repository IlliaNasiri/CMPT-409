from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..types import ArrayLike

# -----------------------------------------------------------------------------
# Optimizer Base Classes
# -----------------------------------------------------------------------------

class OptimizerState(ABC):
    """Base class for optimizer state management"""

    @abstractmethod
    def step(self, w: ArrayLike, X: ArrayLike, y: ArrayLike, lr: float) -> ArrayLike:
        """Take optimization step, return updated weights"""
        pass

    @abstractmethod
    def reset(self):
        """Reset optimizer state (for multi-run experiments)"""
        pass
    
    def __call__(self, w: ArrayLike, X: ArrayLike, y: ArrayLike, lr: float) -> ArrayLike:
        """Allow calling the object like a function"""
        return self.step(w, X, y, lr)

class StatelessOptimizer(OptimizerState):
    """Wrapper for stateless optimizers (GD, SAM, NGD)"""

    def __init__(self, step_fn: Callable[[ArrayLike, ArrayLike, ArrayLike, Optional[float]], ArrayLike]):
        self.step_fn = step_fn

    def step(self, w: ArrayLike, X: ArrayLike, y: ArrayLike, lr: float) -> ArrayLike:
        return self.step_fn(w, X, y, lr)

    def reset(self):
        pass  # No state to reset

class StatefulOptimizer(OptimizerState):
    """Wrapper for optimizers with persistent state (Adam, Adagrad)"""

    def __init__(self, factory_fn: Callable[..., Callable], **kwargs):
        self.factory_fn = factory_fn
        self.kwargs = kwargs
        self.step_fn = None

    def step(self, w: ArrayLike, X: ArrayLike, y: ArrayLike, lr: float) -> ArrayLike:
        if self.step_fn is None:
            # Lazy initialization
            # shape[0] works for both torch.Tensor and np.ndarray
            D = w.shape[0]
            self.step_fn = self.factory_fn(D, lr, **self.kwargs)
        return self.step_fn(w, X, y, lr)

    def reset(self):
        self.step_fn = None  # Force re-initialization

# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def make_optimizer(step_fn: Callable) -> OptimizerState:
    """Create optimizer from stateless step function"""
    return StatelessOptimizer(step_fn)

def make_adaptive_optimizer(factory_fn: Callable, **kwargs) -> OptimizerState:
    """Create optimizer from factory function (Adam, Adagrad)"""
    return StatefulOptimizer(factory_fn, **kwargs)
