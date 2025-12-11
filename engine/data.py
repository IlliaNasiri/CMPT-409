import numpy as np
from sklearn.model_selection import train_test_split
from .types import DatasetSplit
from typing import Dict, Tuple

def make_soudry_dataset(n=200, d=5000, margin=1, sigma=0.3):
    """Generate Soudry-style linearly separable dataset.

    Moved from regression/dataset.py (identical in both modules).
    """
    v = np.ones(d) / np.sqrt(d)
    X_pos = margin * v[None, :] + np.random.randn(n // 2, d) * sigma
    X_neg = -margin * v[None, :] + np.random.randn(n // 2, d) * sigma
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])
    idx = np.random.permutation(n)
    return X[idx], y[idx], v

def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Dict[DatasetSplit, Tuple[np.ndarray, np.ndarray]]:
    """Three-way split: train/val/test"""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state
    )
    return {
        DatasetSplit.Train: (X_train, y_train),
        DatasetSplit.Val: (X_val, y_val),
        DatasetSplit.Test: (X_test, y_test),
    }

def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[DatasetSplit, Tuple[np.ndarray, np.ndarray]]:
    """Backward compatible: train/test only (current behavior)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return {
        DatasetSplit.Train: (X_train, y_train),
        DatasetSplit.Test: (X_test, y_test),
    }
