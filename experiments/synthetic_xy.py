from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataSplit:
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SyntheticSplits:
    train: DataSplit
    learn: DataSplit
    recal: DataSplit
    test: DataSplit


def generate_xy(n: int, seed: int, x_dim: int = 4, y_dim: int = 2) -> DataSplit:
    """
    Synthetic contextual regression problem with 2D outcomes.

    The conditional mean is nonlinear in X and the residual covariance changes
    with X, so a fitted f_hat(X) matters before conformal calibration.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, x_dim))

    mean_cols = [
        0.75 * x[:, 0] + 0.35 * np.sin(x[:, 1]) - 0.20 * x[:, 2],
        -0.45 * x[:, 0] + 0.55 * x[:, 1] ** 2 + 0.30 * x[:, 3],
    ]
    for j in range(2, y_dim):
        mean_cols.append(
            (0.25 + 0.08 * j) * x[:, j % x_dim]
            + 0.20 * np.sin(x[:, (j + 1) % x_dim])
            - 0.12 * x[:, (j + 2) % x_dim] ** 2
        )
    mean = np.column_stack(mean_cols[:y_dim])

    raw = rng.normal(size=(n, y_dim))
    A = rng.normal(size=(y_dim, y_dim))
    cov_root = np.linalg.qr(A)[0] @ np.diag(np.linspace(0.30, 0.08, y_dim))
    scales = 0.80 + 0.45 / (1.0 + np.exp(-x[:, 0:1]))
    eps = scales * (raw @ cov_root.T)

    tail = rng.uniform(size=n) < 0.06
    tail_loc = np.zeros(y_dim)
    tail_loc[0] = 0.55
    if y_dim > 1:
        tail_loc[1] = -0.18
    tail_scale = np.linspace(0.18, 0.08, y_dim)
    eps[tail] += rng.normal(loc=tail_loc, scale=tail_scale, size=(tail.sum(), y_dim))

    return DataSplit(x=x, y=mean + eps)


def make_splits(
    seed: int = 2026,
    n_train: int = 1200,
    n_learn: int = 300,
    n_recal: int = 300,
    n_test: int = 1000,
    y_dim: int = 2,
) -> SyntheticSplits:
    data = generate_xy(n_train + n_learn + n_recal + n_test, seed=seed, y_dim=y_dim)
    i0 = n_train
    i1 = i0 + n_learn
    i2 = i1 + n_recal
    return SyntheticSplits(
        train=DataSplit(data.x[:i0], data.y[:i0]),
        learn=DataSplit(data.x[i0:i1], data.y[i0:i1]),
        recal=DataSplit(data.x[i1:i2], data.y[i1:i2]),
        test=DataSplit(data.x[i2:], data.y[i2:]),
    )
