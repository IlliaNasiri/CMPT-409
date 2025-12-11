import numpy as np
from typing import Dict, Callable, List, Optional, Tuple
from sklearn.svm import LinearSVC
from .types import Metric, MetricKey, DatasetSplit

# Metric computation functions (from regression/metrics.py)
def get_empirical_max_margin(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute max-margin classifier using LinearSVC"""
    svm = LinearSVC(C=1e6, fit_intercept=False, max_iter=100000)
    svm.fit(X, y)
    w = svm.coef_.ravel()
    return w / np.linalg.norm(w)

def get_angle(w: np.ndarray, w_star: np.ndarray) -> float:
    """Angular distance between vectors"""
    w_norm = w / (np.linalg.norm(w) + 1e-12)
    w_star_norm = w_star / (np.linalg.norm(w_star) + 1e-12)
    cos_angle = np.clip(np.dot(w_norm, w_star_norm), -1, 1)
    return float(np.arccos(cos_angle))

def get_direction_distance(w: np.ndarray, w_star: np.ndarray) -> float:
    """L2 distance between normalized directions"""
    w_norm = w / (np.linalg.norm(w) + 1e-12)
    w_star_norm = w_star / (np.linalg.norm(w_star) + 1e-12)
    return float(np.linalg.norm(w_norm - w_star_norm))

def exponential_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Exponential loss: mean(exp(-y * X @ w))"""
    margins = y * (X @ w)
    safe_margins = np.clip(margins, -100, 100)
    return float(np.mean(np.exp(-safe_margins)))

def get_error_rate(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Classification error rate"""
    predictions = np.sign(X @ w)
    return float(np.mean(predictions != y))

class MetricsCollector:
    """Systematically compute all registered metrics on all splits"""

    def __init__(
        self,
        metric_fns: Dict[Metric, Callable],
        w_star: Optional[np.ndarray] = None
    ):
        self.metric_fns = metric_fns
        self.w_star = w_star

    def compute_all(
        self,
        w: np.ndarray,
        datasets: Dict[DatasetSplit, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[MetricKey, float]:
        """Compute ALL metrics on ALL splits systematically"""
        results = {}

        for metric, fn in self.metric_fns.items():
            if metric.requires_reference:
                # Reference metrics (angle, distance): compute once with w_star
                if self.w_star is not None:
                    key = MetricKey(metric, None)
                    results[key] = float(fn(w, self.w_star))
            else:
                # Data-dependent metrics (loss, error): compute per split
                for split, (X, y) in datasets.items():
                    key = MetricKey(metric, split)
                    results[key] = float(fn(w, X, y))

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
