from typing import Dict, Optional, Callable, Tuple, Any, List
from sklearn.svm import LinearSVC
from .types import Metric, MetricKey, DatasetSplit, FloatLike, ArrayLike
from .models import Model
import numpy as np
import torch

def _ensure_consistent_backend(
    primary: ArrayLike,
    reference: ArrayLike
) -> Tuple[Any, Any, Any]:
    """
    Aligns 'reference' to match 'primary' backend/device.
    """
    if isinstance(primary, torch.Tensor):
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference).to(
                device=primary.device, 
                dtype=primary.dtype
            )
        elif isinstance(reference, torch.Tensor) and reference.device != primary.device:
            reference = reference.to(primary.device)
        return primary, reference, torch
    else:
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
        return primary, reference, np

# -----------------------------------------------------------------------------
# 1. Vector Metrics
# -----------------------------------------------------------------------------

def get_empirical_max_margin(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute max-margin classifier using LinearSVC"""
    svm = LinearSVC(C=1e6, fit_intercept=False, max_iter=100000)
    svm.fit(X, y)
    w = svm.coef_.ravel()
    return w / np.linalg.norm(w)

def get_angle(w: ArrayLike, w_star: ArrayLike) -> FloatLike:
    """Returns angle in radians as a scalar (Torch tensor or Numpy scalar)."""
    w, w_star, xp = _ensure_consistent_backend(w, w_star)
    eps = 1e-12
    
    # Operations remain on device
    n_w = xp.linalg.norm(w)
    n_star = xp.linalg.norm(w_star)
    
    if xp == torch:
        dot_val = torch.dot(w.flatten(), w_star.flatten())
        cos_angle = torch.clamp(dot_val / (n_w * n_star + eps), -1.0, 1.0)
        return torch.acos(cos_angle)
    else:
        dot_val = np.dot(w.flatten(), w_star.flatten())
        cos_angle = np.clip(dot_val / (n_w * n_star + eps), -1.0, 1.0)
        return np.arccos(cos_angle)

def get_direction_distance(w: ArrayLike, w_star: ArrayLike) -> FloatLike:
    """L2 distance between normalized directions."""
    w, w_star, xp = _ensure_consistent_backend(w, w_star)
    eps = 1e-12

    w_norm = w / (xp.linalg.norm(w) + eps)
    w_star_norm = w_star / (xp.linalg.norm(w_star) + eps)

    diff = w_norm - w_star_norm

    if xp == torch:
        return torch.norm(diff)
    return np.linalg.norm(diff)

def get_norm(w: ArrayLike, _unused=None) -> FloatLike:
    if isinstance(w, torch.Tensor):
        return torch.norm(w)
    return np.linalg.norm(w)

# -----------------------------------------------------------------------------
# 2. Model Metrics
# -----------------------------------------------------------------------------

def exponential_loss(model: Model, X: ArrayLike, y: ArrayLike) -> FloatLike:
    """Mean Exponential Loss. Returns scalar on device."""
    with torch.no_grad():
        scores = model.forward(X) 
    scores, y, xp = _ensure_consistent_backend(scores, y)
    
    if y.shape != scores.shape:
        y = y.reshape(scores.shape)

    margins = y * scores
    
    if xp == torch:
        safe_margins = torch.clamp(margins, min=-50)
        return torch.mean(torch.exp(-safe_margins))
    else:
        safe_margins = np.clip(margins, -50, 100)
        return np.mean(np.exp(-safe_margins))

def get_error_rate(model: Model, X: ArrayLike, y: ArrayLike) -> FloatLike:
    """Classification Error Rate. Returns scalar on device."""
    with torch.no_grad():
        scores = model.forward(X)
    scores, y, xp = _ensure_consistent_backend(scores, y)
    
    if y.shape != scores.shape:
        y = y.reshape(scores.shape)

    if xp == torch:
        preds = torch.sign(scores)
        return (preds != y).double().mean()
    else:
        preds = np.sign(scores)
        return np.mean(preds != y)

class MetricsCollector:
    """
    Computes metrics without forcing CPU synchronization.
    """
    def __init__(
        self,
        metric_fns: Dict[Metric, Callable[..., FloatLike]],
        w_star: Optional[ArrayLike] = None
    ):
        self.metric_fns = metric_fns
        self.w_star = w_star

    def compute_all(
        self,
        model: Model,
        datasets: Dict[DatasetSplit, Tuple[ArrayLike, ArrayLike]]
    ) -> Dict[MetricKey, FloatLike]:

        results = {}

        # 1. Lazy Fetch of Effective Weights
        w_eff = None
        needs_w_eff = any(m.requires_reference for m in self.metric_fns)

        if needs_w_eff:
            # We grab the reference but DO NOT detach/cpu it unless necessary
            if hasattr(model, "effective_weight"):
                w_eff = model.effective_weight
            elif hasattr(model, "parameters"):
                params = list(model.parameters())
                if len(params) == 1:
                    w_eff = params[0]

            # Flatten only, keep on device
            # if isinstance(w_eff, torch.Tensor):
            #     w_eff = w_eff.view(-1)
            if isinstance(w_eff, np.ndarray):
                w_eff = w_eff.ravel()

        # 2. Compute Metrics
        for metric, fn in self.metric_fns.items():

            if metric.requires_reference:
                if w_eff is None:
                    raise ValueError(f"Needed w_eff to compute the metric {metric.name}")

                key = MetricKey(metric, None)
                if self.w_star is not None:
                    results[key] = fn(w_eff, self.w_star)

            else:
                for split, (X, y) in datasets.items():
                    key = MetricKey(metric, split)
                    results[key] = fn(model, X, y)

        return results


    def get_metric_keys(self, splits: List[DatasetSplit]) -> List[MetricKey]:
        """Get all metric keys that will be computed"""
        keys = []
        for metric in self.metric_fns:
            if metric.requires_reference:
                keys.append(MetricKey(metric, None))
            else:
                for split in splits:
                    keys.append(MetricKey(metric, split))
        return keys
