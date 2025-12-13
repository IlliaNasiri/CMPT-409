from typing import Dict, Optional, Callable
from sklearn.svm import LinearSVC
from .types import Metric, MetricKey, DatasetSplit, ArrayLike
from .models import Model
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Reference Metric Computation (w_star)
# -----------------------------------------------------------------------------


def get_empirical_max_margin(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute max-margin classifier using LinearSVC.

    Args:
        X: Input features (torch.Tensor)
        y: Labels (torch.Tensor)

    Returns:
        w_star: Normalized max-margin weight vector (torch.Tensor)
    """
    # Convert to numpy for sklearn
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    # Compute max-margin classifier
    svm = LinearSVC(C=1e6, fit_intercept=False, max_iter=100000)
    svm.fit(X_np, y_np)
    w = svm.coef_.ravel()
    w_normalized = w / np.linalg.norm(w)

    # Convert back to torch on same device as X
    return torch.tensor(w_normalized, dtype=X.dtype, device=X.device)


# -----------------------------------------------------------------------------
# Vector Metrics (w vs w_star)
# -----------------------------------------------------------------------------


def get_angle(w: torch.Tensor, w_star: torch.Tensor) -> float:
    """
    Compute angle between two vectors in radians.

    Args:
        w: Weight vector
        w_star: Reference weight vector

    Returns:
        Angle in radians (Python float)
    """
    eps = 1e-12
    w_flat = w.flatten()
    w_star_flat = w_star.flatten()

    # Ensure same device
    if w_star.device != w.device:
        w_star_flat = w_star_flat.to(w.device)

    # Compute angle
    dot_val = torch.dot(w_flat, w_star_flat)
    n_w = torch.norm(w_flat)
    n_star = torch.norm(w_star_flat)
    cos_angle = torch.clamp(dot_val / (n_w * n_star + eps), -1.0, 1.0)
    angle = torch.acos(cos_angle)

    return angle.item()


def get_direction_distance(w: torch.Tensor, w_star: torch.Tensor) -> float:
    """
    L2 distance between normalized directions.

    Args:
        w: Weight vector
        w_star: Reference weight vector

    Returns:
        Distance (Python float)
    """
    eps = 1e-12

    # Ensure same device
    if w_star.device != w.device:
        w_star = w_star.to(w.device)

    # Normalize both vectors
    w_norm = w / (torch.norm(w) + eps)
    w_star_norm = w_star / (torch.norm(w_star) + eps)

    # Compute distance
    diff = w_norm - w_star_norm
    distance = torch.norm(diff)

    return distance.item()


def get_norm(w: torch.Tensor, _unused=None) -> float:
    """
    L2 norm of weight vector.

    Args:
        w: Weight vector
        _unused: Unused parameter for API compatibility

    Returns:
        Norm (Python float)
    """
    return torch.norm(w).item()


# -----------------------------------------------------------------------------
# Model Metrics (loss, error)
# -----------------------------------------------------------------------------


def exponential_loss(scores: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute exponential loss: mean(exp(-y * scores))

    Args:
        scores: Model predictions (N,)
        y: Labels {-1, +1} (N,)

    Returns:
        Loss value (Python float)
    """
    margins = y * scores
    safe_margins = torch.clamp(margins, -50, 100)
    loss = torch.mean(torch.exp(-safe_margins))
    return loss.item()


def get_error_rate(scores: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute classification error rate.

    Args:
        scores: Model predictions (N,)
        y: Labels {-1, +1} (N,)

    Returns:
        Error rate in [0, 1] (Python float)
    """
    predictions = torch.sign(scores)
    errors = (predictions != y).float()
    error_rate = torch.mean(errors)
    return error_rate.item()


# -----------------------------------------------------------------------------
# MetricsCollector
# -----------------------------------------------------------------------------


class MetricsCollector:
    """
    Computes multiple metrics on a model without forcing CPU synchronization.
    All computations stay on device until final `.item()` call.
    """

    def __init__(
        self, metric_fns: Dict[Metric, Callable], w_star: Optional[torch.Tensor] = None
    ):
        """
        Args:
            metric_fns: Dict mapping Metric enum to computation function
            w_star: Reference solution for angle/distance metrics (optional)
        """
        self.metric_fns = metric_fns
        self.w_star = w_star

    def compute_all(
        self,
        model: Model,
        datasets: Dict[DatasetSplit, tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[MetricKey, float]:
        """
        Compute all metrics for all dataset splits.

        Args:
            model: Trained model
            datasets: Dict mapping DatasetSplit to (X, y) tuples

        Returns:
            Dict mapping MetricKey to metric values (Python floats)
        """
        results = {}

        # Get model weights (lazily, only if needed)
        w_eff = None

        for metric, metric_fn in self.metric_fns.items():
            if metric.requires_reference:
                # Reference metrics (Angle, Distance): compare w vs w_star
                if self.w_star is None:
                    continue  # Skip if no reference available

                # Lazy fetch of effective weight
                if w_eff is None:
                    w_eff = self._get_effective_weight(model)

                value = metric_fn(w_eff, self.w_star)
                key = MetricKey(metric=metric, split=None)
                results[key] = value
            else:
                # Dataset metrics (Loss, Error): compute on each split
                for split, (X, y) in datasets.items():
                    # Forward pass
                    with torch.no_grad():
                        scores = model.forward(X)

                    # Compute metric
                    value = metric_fn(scores, y)
                    key = MetricKey(metric=metric, split=split)
                    results[key] = value

        return results

    def _get_effective_weight(self, model: Model) -> torch.Tensor:
        """
        Extract effective weight vector from model.
        For linear models, this is just w.
        For multi-layer models, this is the effective linear predictor.
        """
        if hasattr(model, "effective_weight"):
            return model.effective_weight  # type: ignore[attr-defined]
        elif hasattr(model, "w"):
            return model.w  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"Model {type(model).__name__} has no accessible weight vector"
            )

    def get_metric_keys(self, splits: list[DatasetSplit]) -> list[MetricKey]:
        """
        Return list of all metric keys that will be computed.

        Args:
            splits: List of dataset splits available

        Returns:
            List of MetricKey objects
        """
        keys = []
        for metric in self.metric_fns.keys():
            if metric.requires_reference:
                # Reference metrics: one key without split
                keys.append(MetricKey(metric=metric, split=None))
            else:
                # Dataset metrics: one key per split
                for split in splits:
                    keys.append(MetricKey(metric=metric, split=split))
        return keys
