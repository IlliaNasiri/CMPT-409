import numpy as np
import torch
from typing import List, Dict, Union
from .types import MetricKey, ComputeBackend, ArrayLike, FloatLike

class TrainingHistory:
    """Pre-allocated 2D array for metric storage"""

    def __init__(
        self,
        metric_keys: List[MetricKey],
        num_records: int,
        backend: ComputeBackend = ComputeBackend.NumPy,
        device: str = "cpu"
    ):
        self.metric_keys = list(metric_keys)
        self._metric_to_col = {key: i for i, key in enumerate(metric_keys)}
        self._current_idx = 0
        self.backend = backend

        # Pre-allocate single 2D array
        num_metrics = len(metric_keys)

        match backend:
            case ComputeBackend.Torch:
                self._steps = torch.zeros(num_records, dtype=torch.int64, device=device)
                self._data = torch.zeros((num_records, num_metrics), dtype=torch.float64, device=device)
            case ComputeBackend.NumPy:
                self._steps = np.zeros(num_records, dtype=np.int64)
                self._data = np.zeros((num_records, num_metrics), dtype=np.float64)
            case _:
                raise NotImplementedError(f"ComputeBackend of {backend} not implemented")

    def record(self, step: int, metrics: Dict[MetricKey, FloatLike]):
        """Write metrics to pre-allocated row (no append)"""
        idx = self._current_idx
        self._steps[idx] = step
        for key, value in metrics.items():
            col = self._metric_to_col[key]
            self._data[idx, col] = value
        self._current_idx += 1

    def get(self, key: MetricKey) -> ArrayLike:
        """Get column view for a metric (no copy)"""
        col = self._metric_to_col[key]
        return self._data[:self._current_idx, col]

    def get_steps(self) -> ArrayLike:
        """Get recorded step indices"""
        return self._steps[:self._current_idx]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to dict for plotting/saving (converts to numpy)"""
        self.to_cpu()

        steps = self._steps[:self._current_idx]
        data = self._data[:self._current_idx]
        result = {"steps": steps}
        for i, key in enumerate(self.metric_keys):
            result[str(key)] = data[:, i]
        return result

    def to_cpu(self):
        if isinstance(self._steps, torch.Tensor):
            self._steps = self._steps.cpu().numpy()
        if isinstance(self._data, torch.Tensor):
            self._data = self._data.cpu().numpy()
        return self

