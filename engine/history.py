import torch
from typing import List, Dict
from .types import MetricKey

class TrainingHistory:
    """Pre-allocated 2D array for metric storage (PyTorch only)"""

    def __init__(
        self,
        metric_keys: List[MetricKey],
        num_records: int,
        device: str = "cpu"
    ):
        """
        Args:
            metric_keys: List of MetricKey objects to track
            num_records: Maximum number of recording steps
            device: PyTorch device ("cpu" or "cuda:0")
        """
        self.metric_keys = list(metric_keys)
        self._metric_to_col = {key: i for i, key in enumerate(metric_keys)}
        self._current_idx = 0
        self.device = device

        # Pre-allocate single 2D array (Torch only)
        num_metrics = len(metric_keys)
        self._steps = torch.zeros(num_records, dtype=torch.int64, device=device)
        self._data = torch.zeros((num_records, num_metrics), dtype=torch.float64, device=device)

    def record(self, step: int, metrics: Dict[MetricKey, float]):
        """
        Write metrics to pre-allocated row (no append).

        Args:
            step: Training iteration number
            metrics: Dict mapping MetricKey to metric value (Python float)
        """
        idx = self._current_idx
        self._steps[idx] = step
        for key, value in metrics.items():
            col = self._metric_to_col[key]
            self._data[idx, col] = value
        self._current_idx += 1

    def get(self, key: MetricKey) -> torch.Tensor:
        """
        Get column view for a metric (no copy).

        Args:
            key: MetricKey to retrieve

        Returns:
            Tensor of metric values for all recorded steps
        """
        col = self._metric_to_col[key]
        return self._data[:self._current_idx, col]

    def get_steps(self) -> torch.Tensor:
        """
        Get recorded step indices.

        Returns:
            Tensor of step numbers
        """
        return self._steps[:self._current_idx]

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert to dictionary format for plotting/saving.

        Returns:
            Dict with 'steps' and one key per metric
        """
        result = {'steps': self.get_steps()}
        for key in self.metric_keys:
            result[str(key)] = self.get(key)
        return result

    def copy_cpu(self) -> 'TrainingHistory':
        """
        Transfer data from GPU to CPU.

        Returns:
            New TrainingHistory instance with data on CPU
        """
        if self.device == "cpu":
            return self  # Already on CPU

        # Create new history on CPU
        cpu_history = TrainingHistory(
            metric_keys=self.metric_keys,
            num_records=len(self._steps),
            device="cpu"
        )

        # Copy data to CPU
        cpu_history._steps = self._steps.cpu()
        cpu_history._data = self._data.cpu()
        cpu_history._current_idx = self._current_idx

        return cpu_history
