import numpy as np
import torch
from sklearn.model_selection import train_test_split
from .types import DatasetSplit
from typing import Dict, Tuple

def make_soudry_dataset(n=200, d=5000, margin=1, sigma=0.3, device="cpu"):
    """Generate Soudry-style linearly separable dataset.

    Returns PyTorch tensors (all training uses Torch now).

    Args:
        n: Number of samples
        d: Dimension
        margin: Separation margin
        sigma: Noise level
        device: PyTorch device ("cpu" or "cuda:0")

    Returns:
        X: Input features (n, d) - torch.Tensor
        y: Labels {-1, +1} (n,) - torch.Tensor
        v: Population direction (d,) - torch.Tensor
    """
    v = np.ones(d) / np.sqrt(d)
    X_pos = margin * v[None, :] + np.random.randn(n // 2, d) * sigma
    X_neg = -margin * v[None, :] + np.random.randn(n // 2, d) * sigma
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])
    idx = np.random.permutation(n)

    # Convert to Torch tensors
    X_torch = torch.tensor(X[idx], dtype=torch.float64, device=device)
    y_torch = torch.tensor(y[idx], dtype=torch.float64, device=device)
    v_torch = torch.tensor(v, dtype=torch.float64, device=device)

    return X_torch, y_torch, v_torch

def split_train_val_test(
    X: torch.Tensor,
    y: torch.Tensor,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42
) -> Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]]:
    """Three-way split: train/val/test

    Args:
        X: Input features (torch.Tensor)
        y: Labels (torch.Tensor)
        val_size: Validation set fraction
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Dict mapping DatasetSplit to (X, y) tuples (all torch.Tensor)
    """
    # Convert to numpy for sklearn, then back to torch
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    device = X.device
    dtype = X.dtype

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, random_state=random_state
    )

    return {
        DatasetSplit.Train: (
            torch.tensor(X_train, dtype=dtype, device=device),
            torch.tensor(y_train, dtype=dtype, device=device)
        ),
        DatasetSplit.Val: (
            torch.tensor(X_val, dtype=dtype, device=device),
            torch.tensor(y_val, dtype=dtype, device=device)
        ),
        DatasetSplit.Test: (
            torch.tensor(X_test, dtype=dtype, device=device),
            torch.tensor(y_test, dtype=dtype, device=device)
        ),
    }

def split_train_test(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[DatasetSplit, Tuple[torch.Tensor, torch.Tensor]]:
    """Backward compatible: train/test only (current behavior)

    Args:
        X: Input features (torch.Tensor)
        y: Labels (torch.Tensor)
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Dict mapping DatasetSplit to (X, y) tuples (all torch.Tensor)
    """
    # Convert to numpy for sklearn, then back to torch
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()
    device = X.device
    dtype = X.dtype

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )

    return {
        DatasetSplit.Train: (
            torch.tensor(X_train, dtype=dtype, device=device),
            torch.tensor(y_train, dtype=dtype, device=device)
        ),
        DatasetSplit.Test: (
            torch.tensor(X_test, dtype=dtype, device=device),
            torch.tensor(y_test, dtype=dtype, device=device)
        ),
    }
