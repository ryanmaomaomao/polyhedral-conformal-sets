from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from itertools import combinations

import numpy as np
from math import gamma, pi


@dataclass
class SetResult:
    name: str
    kind: str
    color: str
    W: np.ndarray | None = None
    h: np.ndarray | None = None
    radius: float | None = None
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None
    threshold: float | None = None
    support_fn: Callable[[np.ndarray], float] | None = None
    support_argmax_fn: Callable[[np.ndarray], np.ndarray] | None = None
    robust_solver_fn: Callable[[np.ndarray, np.ndarray, object], object] | None = None
    chol_factor: np.ndarray | None = None
    volume_fn: Callable[[], float] | None = None
    coverage_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    metadata: dict = field(default_factory=dict)

    def contains_residuals(self, r: np.ndarray) -> np.ndarray:
        if self.score_fn is not None and self.threshold is not None:
            return self.score_fn(r) <= self.threshold + 1e-9
        if self.W is not None and self.h is not None:
            return np.all(r @ self.W.T <= self.h[None, :] + 1e-9, axis=1)
        if self.kind == "ball" and self.radius is not None:
            return np.linalg.norm(r, axis=1) <= self.radius + 1e-9
        raise NotImplementedError(f"Containment not implemented for {self.name}.")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    k = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    k = min(max(k, 1), len(scores))
    return float(np.sort(scores)[k - 1])


def directions(k: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def ball_as_halfspaces(radius: float, k: int = 96) -> tuple[np.ndarray, np.ndarray]:
    W = directions(k)
    return W, radius * np.ones(k)


def box_halfspaces(radius: float, dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    W = np.vstack([np.eye(dim), -np.eye(dim)])
    return W, radius * np.ones(2 * dim)


def polytope_vertices(W: np.ndarray, h: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Enumerate vertices of a bounded 2D polytope {r: W r <= h}."""
    W = np.asarray(W, dtype=float)
    h = np.asarray(h, dtype=float)
    if W.shape[1] != 2:
        raise ValueError("Vertex enumeration is currently implemented only for 2D.")

    vertices = []
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[0]):
            A = np.vstack([W[i], W[j]])
            if abs(np.linalg.det(A)) <= tol:
                continue
            candidate = np.linalg.solve(A, np.array([h[i], h[j]]))
            if np.all(W @ candidate <= h + 1e-7):
                vertices.append(candidate)

    if not vertices:
        raise ValueError("No vertices found. Check that the polytope is bounded and nonempty.")

    vertices = np.unique(np.round(np.vstack(vertices), 10), axis=0)
    center = vertices.mean(axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
    return vertices[order]


def polytope_vertices_nd(W: np.ndarray, h: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Enumerate vertices of a bounded low-dimensional polytope {r: W r <= h}.

    This is intended for the small d,K hyperparameter sweeps, not large-scale
    vertex enumeration.
    """
    W = np.asarray(W, dtype=float)
    h = np.asarray(h, dtype=float)
    d = W.shape[1]
    if d == 2:
        return polytope_vertices(W, h, tol=tol)

    vertices = []
    for idx in combinations(range(W.shape[0]), d):
        A = W[list(idx)]
        if abs(np.linalg.det(A)) <= tol:
            continue
        candidate = np.linalg.solve(A, h[list(idx)])
        if np.all(W @ candidate <= h + 1e-7):
            vertices.append(candidate)
    if not vertices:
        raise ValueError("No vertices found. Check that the polytope is bounded and nonempty.")
    return np.unique(np.round(np.vstack(vertices), 10), axis=0)


def polygon_area(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def set_area(set_result: SetResult) -> float:
    if set_result.volume_fn is not None:
        return set_result.volume_fn()
    if set_result.kind == "ball" and set_result.radius is not None:
        dim = 2
        if set_result.W is not None:
            dim = int(set_result.W.shape[1])
        elif set_result.metadata and "dim" in set_result.metadata:
            dim = int(set_result.metadata["dim"])
        return float(pi ** (dim / 2.0) / gamma(dim / 2.0 + 1.0) * set_result.radius**dim)
    if set_result.kind == "ellipse" and set_result.threshold is not None and set_result.chol_factor is not None:
        return float(np.pi * set_result.threshold**2 * abs(np.linalg.det(set_result.chol_factor)))
    if set_result.W is None or set_result.h is None:
        raise NotImplementedError(f"Area not implemented for {set_result.name}.")
    if set_result.W.shape[1] != 2:
        try:
            from scipy.spatial import ConvexHull

            verts = polytope_vertices_nd(set_result.W, set_result.h)
            if len(verts) <= set_result.W.shape[1]:
                return 0.0
            hull = ConvexHull(verts)
            return float(hull.volume)
        except Exception:
            return float("nan")
    return polygon_area(polytope_vertices(set_result.W, set_result.h))
