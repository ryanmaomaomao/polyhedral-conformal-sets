from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeFeaturePredictor:
    coef: np.ndarray
    ridge: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return design_matrix(x) @ self.coef


def design_matrix(x: np.ndarray) -> np.ndarray:
    """Small nonlinear feature map for f_hat(X)."""
    return np.column_stack(
        [
            np.ones(len(x)),
            x,
            x[:, 0] ** 2,
            x[:, 1] ** 2,
            np.sin(x[:, 0]),
            np.sin(x[:, 1]),
        ]
    )


def fit_ridge_predictor(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3) -> RidgeFeaturePredictor:
    phi = design_matrix(x)
    gram = phi.T @ phi + ridge * np.eye(phi.shape[1])
    coef = np.linalg.solve(gram, phi.T @ y)
    return RidgeFeaturePredictor(coef=coef, ridge=ridge)


def residuals(model: RidgeFeaturePredictor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - model.predict(x)

