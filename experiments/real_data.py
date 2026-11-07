from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from .synthetic_xy import DataSplit, SyntheticSplits


ENERGY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
ENERGY_FILE = "ENB2012_data.xlsx"


def _standardize_with_train(
    train: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    loc = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return tuple((arr - loc) / scale for arr in arrays)


def load_energy_efficiency(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UCI Energy Efficiency with X in R^8 and Y=(heating,cooling)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ENERGY_FILE
    if not path.exists():
        urlretrieve(ENERGY_URL, path)

    frame = pd.read_excel(path, engine="openpyxl")
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if numeric.shape[1] < 10:
        raise ValueError(f"Expected at least 10 numeric columns in {path}, found {numeric.shape[1]}.")

    x = numeric.iloc[:, :8].to_numpy(dtype=float)
    y = numeric.iloc[:, -2:].to_numpy(dtype=float)
    return x, y


def make_energy_splits(
    seed: int,
    cache_dir: Path,
    n_train: int = 320,
    n_learn: int = 160,
    n_recal: int = 144,
    n_test: int = 144,
    standardize: bool = True,
) -> SyntheticSplits:
    x, y = load_energy_efficiency(cache_dir)
    n_needed = n_train + n_learn + n_recal + n_test
    if n_needed > len(x):
        raise ValueError(f"Requested {n_needed} observations but Energy only has {len(x)} rows.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))[:n_needed]
    x = x[order]
    y = y[order]

    i0 = n_train
    i1 = i0 + n_learn
    i2 = i1 + n_recal
    x_train, x_learn, x_recal, x_test = x[:i0], x[i0:i1], x[i1:i2], x[i2:]
    y_train, y_learn, y_recal, y_test = y[:i0], y[i0:i1], y[i1:i2], y[i2:]

    if standardize:
        x_train, x_learn, x_recal, x_test = _standardize_with_train(
            x_train, x_train, x_learn, x_recal, x_test
        )
        y_train, y_learn, y_recal, y_test = _standardize_with_train(
            y_train, y_train, y_learn, y_recal, y_test
        )

    return SyntheticSplits(
        train=DataSplit(x_train, y_train),
        learn=DataSplit(x_learn, y_learn),
        recal=DataSplit(x_recal, y_recal),
        test=DataSplit(x_test, y_test),
    )
