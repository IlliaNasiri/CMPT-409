"""Unified optimization experiment framework"""

# Core types
from .types import (
    ComputeBackend,
    DatasetSplit,
    Metric,
    Optimizer,
    MetricKey,
)

# Data management
from .data import (
    make_soudry_dataset,
    split_train_val_test,
    split_train_test,
)

# Metrics
from .metrics import (
    MetricsCollector,
    get_empirical_max_margin,
    get_angle,
    get_direction_distance,
    exponential_loss,
    get_error_rate,
)

# History
from .history import TrainingHistory

# Models
from .models import LinearModel, TwoLayerModel

# Training
from .trainer import run_training

__all__ = [
    # Types
    'ComputeBackend',
    'DatasetSplit',
    'Metric',
    'Optimizer',
    'MetricKey',
    # Data
    'make_soudry_dataset',
    'split_train_val_test',
    'split_train_test',
    # Metrics
    'MetricsCollector',
    'get_empirical_max_margin',
    'get_angle',
    'get_direction_distance',
    'exponential_loss',
    'get_error_rate',
    # History
    'TrainingHistory',
    # Models
    'LinearModel',
    'TwoLayerModel',
    # Training
    'run_training',
]
