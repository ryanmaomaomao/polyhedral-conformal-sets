from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sets import polytope_vertices


@dataclass(frozen=True)
class LossSpec:
    name: str = "linear"
    quadratic_weight: float = 0.0
    # Newsvendor overage / underage costs (per component).  None means use defaults.
    newsvendor_c_o: tuple[float, ...] | None = None
    newsvendor_c_u: tuple[float, ...] | None = None

    def _nv_costs(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        c_o = np.asarray(self.newsvendor_c_o if self.newsvendor_c_o is not None else [0.3] * d, dtype=float)
        c_u = np.asarray(self.newsvendor_c_u if self.newsvendor_c_u is not None else [0.7] * d, dtype=float)
        return c_o, c_u

    def penalty(self, z: np.ndarray) -> float:
        if self.quadratic_weight <= 0.0:
            return 0.0
        return float(self.quadratic_weight * (z[0] - z[1]) ** 2)

    def loss(self, y: np.ndarray, cost: np.ndarray, z: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        cost = np.asarray(cost, dtype=float)
        if self.name == "newsvendor":
            c_o, c_u = self._nv_costs(len(z))
            overage = c_o * np.maximum(z - y, 0.0)
            underage = c_u * np.maximum(y - z, 0.0)
            return float(np.sum(overage + underage))
        return float((cost + y) @ z + self.penalty(z))

    def hindsight_loss(self, y: np.ndarray, cost: np.ndarray) -> float:
        """Best achievable loss after observing an outcome and choosing z in the simplex."""
        y = np.asarray(y, dtype=float)
        cost = np.asarray(cost, dtype=float)
        if self.name == "newsvendor":
            d = len(y)
            c_o, c_u = self._nv_costs(d)
            if d == 2:
                # min_{t in [0,1]} g((t,1-t), y): piecewise linear, evaluate at breakpoints
                def nv2(t: float) -> float:
                    z = np.array([t, 1.0 - t])
                    return self.loss(y, cost, z)
                breakpoints = [0.0, 1.0, float(np.clip(y[0], 0.0, 1.0)), float(np.clip(1.0 - y[1], 0.0, 1.0))]
                return min(nv2(t) for t in breakpoints)
            # Generic: scipy minimize over simplex
            from scipy.optimize import minimize
            def obj(z: np.ndarray) -> float:
                return self.loss(y, cost, z)
            res = minimize(obj, np.ones(d) / d, method="SLSQP",
                           bounds=[(0.0, 1.0)] * d,
                           constraints=[{"type": "eq", "fun": lambda z: float(np.sum(z) - 1.0)}],
                           options={"maxiter": 200})
            return float(res.fun) if res.success else self.loss(y, cost, np.ones(d) / d)
        a = cost + y
        if self.quadratic_weight <= 0.0:
            return float(np.min(a))
        if len(a) != 2:
            return float(np.min(a))
        beta = float(self.quadratic_weight)
        d_val = float(a[0] - a[1])
        t = float(np.clip((4.0 * beta - d_val) / (8.0 * beta), 0.0, 1.0))
        z = np.array([t, 1.0 - t])
        return self.loss(y, cost, z)


@dataclass
class RobustResult:
    z: np.ndarray
    worst_y: np.ndarray
    objective: float
    worst_residual: np.ndarray


def _linear_candidates(vertices: np.ndarray) -> list[float]:
    slopes = vertices[:, 0] - vertices[:, 1]
    intercepts = vertices[:, 1]

    candidates = [0.0, 1.0]
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            denom = slopes[i] - slopes[j]
            if abs(denom) < 1e-12:
                continue
            t = (intercepts[j] - intercepts[i]) / denom
            if -1e-10 <= t <= 1.0 + 1e-10:
                candidates.append(float(np.clip(t, 0.0, 1.0)))
    return candidates


def newsvendor_robust_via_support(
    support_fn,
    support_argmax_fn,
    center: np.ndarray,
    cost: np.ndarray,
    loss_spec: LossSpec,
) -> RobustResult:
    """Exact 2D newsvendor robust decision over a convex set given its support function.

    The newsvendor loss is a max of 4 affine functions of y (overage/underage per
    coordinate): a_s^T y + b_s(z).  Hence max_{y in center + C} loss(y, z) =
    max_s [a_s^T center + sigma_C(a_s) + b_s(z)], which is convex piecewise-linear
    in t for z = (t, 1 - t); its minimum is attained at t in {0, 1} or at a
    crossing of two pieces.  (The earlier candidate list t = v_0, 1 - v_1 over
    vertices v ignored the center and the crossings between vertices, and could
    return a suboptimal decision.)
    """
    from itertools import product

    center = np.asarray(center, dtype=float)
    c_o, c_u = loss_spec._nv_costs(2)
    pieces = []
    for pattern in product([0, 1], repeat=2):  # 0 = overage piece, 1 = underage piece
        a = np.array([-c_o[j] if p == 0 else c_u[j] for j, p in enumerate(pattern)])
        zc = np.array([c_o[j] if p == 0 else -c_u[j] for j, p in enumerate(pattern)])
        pieces.append((a, zc, float(a @ center + support_fn(a))))

    alphas = np.array([zc[0] - zc[1] for _, zc, _ in pieces])
    betas = np.array([zc[1] + const for _, zc, const in pieces])

    def value(t: float) -> float:
        return float(np.max(alphas * t + betas))

    candidates = [0.0, 1.0]
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if abs(alphas[i] - alphas[j]) > 1e-12:
                t = (betas[j] - betas[i]) / (alphas[i] - alphas[j])
                if 0.0 <= t <= 1.0:
                    candidates.append(float(t))
    best_t = min(candidates, key=value)
    z = np.array([best_t, 1.0 - best_t])
    s_star = int(np.argmax(alphas * best_t + betas))
    worst_residual = np.asarray(support_argmax_fn(pieces[s_star][0]), dtype=float)
    return RobustResult(
        z=z,
        worst_y=center + worst_residual,
        objective=value(best_t),
        worst_residual=worst_residual,
    )


def solve_robust_decision(
    set_result,
    center: np.ndarray,
    cost: np.ndarray,
    loss_spec: LossSpec | None = None,
) -> RobustResult:
    """
    Solve min_{z in simplex} max_{r in C} loss(center + r, z).

    The experiment is intentionally 2D.  With z=(t,1-t), the linear loss is
    solved exactly by checking endpoints and vertex-line intersections.  For
    the quadratic-regularized loss we minimize the resulting 1D convex function
    by golden-section search and keep the same worst-case residual semantics.
    """
    loss_spec = loss_spec or LossSpec()
    center = np.asarray(center, dtype=float)
    cost = np.asarray(cost, dtype=float)
    if getattr(set_result, "robust_solver_fn", None) is not None:
        return set_result.robust_solver_fn(center, cost, loss_spec)
    if center.shape != cost.shape:
        raise ValueError("center and cost must have the same dimension.")
    if center.shape != (2,):
        return _solve_robust_decision_generic(set_result, center, cost, loss_spec)
    if set_result.W is None or set_result.h is None:
        raise ValueError(f"{set_result.name} does not expose a polyhedral representation.")

    vertices = polytope_vertices(set_result.W, set_result.h)
    base = center + cost

    if loss_spec.name == "newsvendor":
        return newsvendor_robust_via_support(
            lambda a: float(np.max(vertices @ a)),
            lambda a: vertices[int(np.argmax(vertices @ a))],
            center,
            cost,
            loss_spec,
        )

    def objective_at(t: float) -> float:
        z = np.array([t, 1.0 - t])
        return float(base @ z + np.max(vertices @ z) + loss_spec.penalty(z))

    candidates = _linear_candidates(vertices)
    if loss_spec.quadratic_weight > 0.0:
        lo, hi = 0.0, 1.0
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        fc = objective_at(c)
        fd = objective_at(d)
        for _ in range(90):
            if fc <= fd:
                hi = d
                d = c
                fd = fc
                c = hi - inv_phi * (hi - lo)
                fc = objective_at(c)
            else:
                lo = c
                c = d
                fc = fd
                d = lo + inv_phi * (hi - lo)
                fd = objective_at(d)
        candidates.append(0.5 * (lo + hi))

    best_t = min(candidates, key=objective_at)
    z = np.array([best_t, 1.0 - best_t])
    worst_idx = int(np.argmax(vertices @ z))
    worst_residual = vertices[worst_idx]
    worst_y = center + worst_residual
    return RobustResult(
        z=z,
        worst_y=worst_y,
        objective=loss_spec.loss(worst_y, cost, z),
        worst_residual=worst_residual,
    )


def _solve_robust_decision_generic(set_result, center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
    """Generic d-dimensional solver.

    For the newsvendor loss, uses vertex enumeration on the polytope (convex
    maximization over a polytope is attained at a vertex).  For linear losses
    uses scipy linprog support.  Quadratic regularization is kept only for
    the 2D specialized path.
    """
    from scipy.optimize import linprog, minimize

    d = len(center)
    center = np.asarray(center, dtype=float)
    cost = np.asarray(cost, dtype=float)
    base = center + cost

    if loss_spec.name == "newsvendor":
        if set_result.W is None or set_result.h is None:
            raise ValueError(f"{set_result.name} does not expose a polyhedral representation for newsvendor.")
        verts = polytope_vertices(set_result.W, set_result.h)

        def total_loss(z_arr: np.ndarray) -> float:
            return max(loss_spec.loss(center + v, cost, z_arr) for v in verts)

        # total_loss is a max of affine functions of z, hence convex on the
        # simplex: a single start suffices and a much smaller iteration
        # budget still converges, which matters because this generic d>2
        # path is called many times per fit (per candidate x per center).
        res = minimize(
            total_loss, x0=np.ones(d) / d, method="SLSQP",
            bounds=[(0.0, 1.0)] * d,
            constraints=[{"type": "eq", "fun": lambda z: float(np.sum(z) - 1.0)}],
            options={"maxiter": 50},
        )
        best = (float(res.fun), np.asarray(res.x, dtype=float)) if res.success else None
        z = np.maximum(best[1], 0.0) if best else np.ones(d) / d
        z = z / max(float(np.sum(z)), 1e-12)
        worst_idx = int(np.argmax([loss_spec.loss(center + v, cost, z) for v in verts]))
        worst_residual = verts[worst_idx]
        worst_y = center + worst_residual
        return RobustResult(z=z, worst_y=worst_y, objective=loss_spec.loss(worst_y, cost, z), worst_residual=worst_residual)

    if loss_spec.quadratic_weight > 0.0:
        # The experiment sweep allows d>2, but the quadratic regularizer in the
        # paper draft is 2D-specific. Keep a well-defined linear comparison.
        loss_spec = LossSpec(name=loss_spec.name, quadratic_weight=0.0)

    def support(z: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.asarray(z, dtype=float)
        if set_result.kind == "ball" and set_result.radius is not None:
            norm = float(np.linalg.norm(z))
            if norm <= 1e-12:
                r = np.zeros(d)
            else:
                r = float(set_result.radius) * z / norm
            return float(z @ r), r
        if set_result.W is None or set_result.h is None:
            raise ValueError(f"{set_result.name} does not expose a supported representation.")
        res = linprog(
            c=-z,
            A_ub=np.asarray(set_result.W, dtype=float),
            b_ub=np.asarray(set_result.h, dtype=float),
            bounds=[(None, None)] * d,
            method="highs",
        )
        if res.status != 0:
            raise RuntimeError(f"Support LP failed for {set_result.name}: {res.message}")
        r = np.asarray(res.x, dtype=float)
        return float(z @ r), r

    def objective(z: np.ndarray) -> float:
        supp, _ = support(z)
        return float(base @ z + supp)

    constraints = [{"type": "eq", "fun": lambda z: float(np.sum(z) - 1.0)}]
    bounds = [(0.0, 1.0)] * d
    # objective(z) = base^T z + support(z) is convex (a support function plus
    # a linear term) on the simplex, so a single start suffices; this path is
    # called many times per fit (per candidate x per center), so the smaller
    # iteration budget matters a lot for wall-clock time.
    res = minimize(objective, x0=np.ones(d) / d, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 50})
    best = (float(res.fun), np.asarray(res.x, dtype=float)) if res.success else None
    if best is None:
        z = np.ones(d) / d
    else:
        z = np.maximum(best[1], 0.0)
        z = z / max(float(np.sum(z)), 1e-12)
    _, worst_residual = support(z)
    worst_y = center + worst_residual
    return RobustResult(
        z=z,
        worst_y=worst_y,
        objective=loss_spec.loss(worst_y, cost, z),
        worst_residual=worst_residual,
    )


def solve_linear_robust_decision(set_result, center: np.ndarray, cost: np.ndarray) -> RobustResult:
    return solve_robust_decision(set_result, center=center, cost=cost, loss_spec=LossSpec())


def empirical_coverage(
    set_result,
    residuals: np.ndarray,
    centers: np.ndarray | None = None,
) -> float:
    if getattr(set_result, "coverage_fn", None) is not None and centers is not None:
        residuals = np.asarray(residuals, dtype=float)
        centers = np.asarray(centers, dtype=float)
        n = min(len(residuals), len(centers))
        if n == 0:
            return float("nan")
        return float(np.mean(set_result.coverage_fn(centers[:n], residuals[:n])))
    return float(np.mean(set_result.contains_residuals(residuals)))


def average_robust_metrics(
    set_result,
    centers: np.ndarray,
    y_true: np.ndarray,
    cost: np.ndarray,
    loss_spec: LossSpec | None = None,
    max_points: int | None = None,
) -> dict[str, float]:
    loss_spec = loss_spec or LossSpec()
    if max_points is not None:
        centers = centers[:max_points]
        y_true = y_true[:max_points]

    robust_objectives = []
    realized_losses = []
    realized_regrets = []
    z0 = []
    worst_y = []

    for center, y in zip(centers, y_true):
        result = solve_robust_decision(set_result, center=center, cost=cost, loss_spec=loss_spec)
        realized_loss = loss_spec.loss(y, cost, result.z)
        hindsight_loss = loss_spec.hindsight_loss(y, cost)
        robust_objectives.append(result.objective)
        realized_losses.append(realized_loss)
        realized_regrets.append(realized_loss - hindsight_loss)
        z0.append(result.z[0])
        worst_y.append(result.worst_y)

    worst_y_arr = np.vstack(worst_y)
    return {
        "avg_robust_objective": float(np.mean(robust_objectives)),
        "avg_realized_loss": float(np.mean(realized_losses)),
        "avg_realized_regret": float(np.mean(realized_regrets)),
        "avg_z0": float(np.mean(z0)),
        "avg_worst_y0": float(np.mean(worst_y_arr[:, 0])),
        "avg_worst_y1": float(np.mean(worst_y_arr[:, 1])),
        "n_robust_eval": float(len(robust_objectives)),
    }
