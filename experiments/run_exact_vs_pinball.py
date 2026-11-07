"""Exact MIQCP versus stochastic pinball-subgradient diagnostics.

This module is separate from the main-table pipeline (``run_synthetic2d.py``).
It runs two controlled diagnostics:

1. A small-instance comparison of an exact empirical-quantile MIQCP
   formulation and the stochastic pinball-subgradient method over the same
   compact, normalized, positively spanning geometry class.
2. A matched-cardinality sensitivity study for random and structured finite
   libraries.

The MIQCP is an exact *formulation*.  A numerical run is marked certified only
when Gurobi returns ``OPTIMAL`` with a reported global relative gap no larger
than the requested tolerance.
Time-limited incumbents remain useful diagnostics but are never called exact
solutions.
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from experiments.baselines import (
    BaselineContext,
    _cost,
    _logsumexp_scores,
    _random_cut_templates,
    _random_templates,
    _select_min_robust,
    _set_from_template,
    _smooth_pinball_loss,
    _template_grid,
    _wc_objective_fixed_radius,
)
from experiments.predictors import fit_ridge_predictor, residuals
from experiments.robust import LossSpec, average_robust_metrics, empirical_coverage
from experiments.sets import SetResult, conformal_quantile, set_area
from experiments.synthetic_xy import make_splits


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_COST = np.array([-0.08, 0.0])
LINEAR_LOSS = LossSpec(name="linear")


@dataclass
class PreparedData:
    r_learn: np.ndarray
    r_recal: np.ndarray
    r_test: np.ndarray
    centers_learn: np.ndarray
    centers_test: np.ndarray
    y_test: np.ndarray


@dataclass
class GeometryFit:
    learner: str
    W: np.ndarray | None
    train_radius: float
    wall_time: float
    status: str
    optimizer_time: float = float("nan")
    incumbent: float = float("nan")
    best_bound: float = float("nan")
    gap_abs: float = float("nan")
    gap_rel: float = float("nan")
    certified: bool = False
    node_count: float = float("nan")
    work: float = float("nan")
    solution_count: int = 0
    n_variables: int = 0
    n_binary: int = 0
    n_constraints: int = 0
    raw_surrogate_objective: float = float("nan")


def _prepare_data(
    seed: int,
    n_learn: int,
    n_recal: int,
    n_test: int,
    n_learn_pool: int | None = None,
) -> PreparedData:
    """Prepare one synthetic-linear split.

    When ``n_learn_pool`` is supplied, every value of ``n_learn`` is a prefix
    of the same learning pool.  This keeps the training sample, predictor,
    synthetic covariance realization, re-calibration sample, and test sample
    fixed across the sample-size sweep.
    """
    pool_size = int(n_learn if n_learn_pool is None else n_learn_pool)
    if n_learn > pool_size:
        raise ValueError("n_learn cannot exceed n_learn_pool.")
    splits = make_splits(
        seed=seed,
        n_train=1200,
        n_learn=pool_size,
        n_recal=n_recal,
        n_test=n_test,
        y_dim=2,
    )
    predictor = fit_ridge_predictor(splits.train.x, splits.train.y)
    x_learn = splits.learn.x[:n_learn]
    y_learn = splits.learn.y[:n_learn]
    return PreparedData(
        r_learn=residuals(predictor, x_learn, y_learn),
        r_recal=residuals(predictor, splits.recal.x, splits.recal.y),
        r_test=residuals(predictor, splits.test.x, splits.test.y),
        centers_learn=predictor.predict(x_learn),
        centers_test=predictor.predict(splits.test.x),
        y_test=splits.test.y,
    )


def _paired_spd_matrix(phi: float, psi: float) -> np.ndarray:
    """Return A with A=A^T>0 and 2||A||_F^2=1.

    For W=(e1,e2,-e1,-e2)^T A, this gives ||W||_F=1 and paired,
    positively spanning normals.  The eigenvalues are parameterized on the
    circle a^2+b^2=1/2.
    """
    scale = 1.0 / math.sqrt(2.0)
    eig = np.array([scale * math.cos(psi), scale * math.sin(psi)])
    c, s = math.cos(phi), math.sin(phi)
    rotation = np.array([[c, -s], [s, c]])
    return rotation @ np.diag(eig) @ rotation.T


def _paired_normals(A: np.ndarray) -> np.ndarray:
    return np.vstack([A[0], A[1], -A[0], -A[1]])


def _exact_learning_objective(W: np.ndarray, r_learn: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    scores = np.max(r_learn @ W.T, axis=1)
    radius = conformal_quantile(scores, alpha=0.10)
    objective = _wc_objective_fixed_radius(
        W,
        radius,
        center[None, :],
        DEFAULT_COST,
        LINEAR_LOSS,
    )
    return float(objective), float(radius)


def _solve_exact_miqcp(
    r_learn: np.ndarray,
    center: np.ndarray,
    seed: int,
    time_limit: float,
    mip_gap: float,
    eig_lb: float,
    threads: int,
) -> GeometryFit:
    """Solve the exact empirical-quantile problem on a common SPD-paired class.

    The class is
        W=(a_1,a_2,-a_1,-a_2)^T, A=[a_1;a_2]=A^T,
        ||W||_F=1, lambda_min(A)>=eig_lb, b=0.
    It is a compact subset of the revised Theta_{eta,0}; in particular,
    conv{w_k} contains an Euclidean ball of radius at least eig_lb/sqrt(2).
    """
    total_t0 = time.perf_counter()
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("The exact diagnostic requires gurobipy and a Gurobi license.") from exc

    r_learn = np.asarray(r_learn, dtype=float)
    n, d = r_learn.shape
    if d != 2:
        raise ValueError("The controlled exact diagnostic is defined for d=2.")

    alpha = 0.10
    kappa = min(int(math.ceil((n + 1) * (1.0 - alpha))), n)
    residual_bound = float(np.max(np.linalg.norm(r_learn, axis=1)))
    q_ub = residual_bound / math.sqrt(2.0) + 1e-8
    big_m = q_ub
    dual_ub = 1.0 / eig_lb

    model = gp.Model("exact_quantile_miqcp")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = float(mip_gap)
    model.Params.MIPGapAbs = 1e-5
    model.Params.Threads = int(threads)
    model.Params.Seed = int(seed)

    # A is symmetric.  Its paired rows define W.
    a00 = model.addVar(lb=eig_lb, ub=1.0 / math.sqrt(2.0), name="a00")
    a01 = model.addVar(lb=-0.5, ub=0.5, name="a01")
    a11 = model.addVar(lb=eig_lb, ub=1.0 / math.sqrt(2.0), name="a11")
    q = model.addVar(lb=0.0, ub=q_ub, name="q")
    covered = model.addVars(n, vtype=GRB.BINARY, name="covered")
    z = model.addVars(2, lb=0.0, ub=1.0, name="z")
    lam = model.addVars(4, lb=0.0, ub=dual_ub, name="lambda")

    # ||W||_F=1 because W contains the two rows of A and their negatives.
    model.addQConstr(
        2.0 * (a00 * a00 + 2.0 * a01 * a01 + a11 * a11) == 1.0,
        name="frobenius_normalization",
    )
    # A-eig_lb I is positive semidefinite (the diagonal constraints are bounds).
    model.addQConstr(
        (a00 - eig_lb) * (a11 - eig_lb) >= a01 * a01,
        name="interior_margin",
    )

    # W^T lambda=z.  With A symmetric and paired rows, the valid dual bound
    # follows from ||A^{-1}z||_2 <= 1/eig_lb.
    model.addQConstr(
        a00 * (lam[0] - lam[2]) + a01 * (lam[1] - lam[3]) == z[0],
        name="dual_balance_0",
    )
    model.addQConstr(
        a01 * (lam[0] - lam[2]) + a11 * (lam[1] - lam[3]) == z[1],
        name="dual_balance_1",
    )
    model.addConstr(z[0] + z[1] == 1.0, name="simplex")
    model.addConstr(gp.quicksum(covered[i] for i in range(n)) >= kappa, name="order_statistic")

    for i, eps in enumerate(r_learn):
        v0 = float(eps[0]) * a00 + float(eps[1]) * a01
        v1 = float(eps[0]) * a01 + float(eps[1]) * a11
        rhs = q + big_m * (1.0 - covered[i])
        model.addConstr(v0 <= rhs, name=f"score_pos0_{i}")
        model.addConstr(v1 <= rhs, name=f"score_pos1_{i}")
        model.addConstr(-v0 <= rhs, name=f"score_neg0_{i}")
        model.addConstr(-v1 <= rhs, name=f"score_neg1_{i}")

    base = DEFAULT_COST + np.asarray(center, dtype=float)
    objective = float(base[0]) * z[0] + float(base[1]) * z[1]
    objective += q * gp.quicksum(lam[k] for k in range(4))
    model.setObjective(objective, GRB.MINIMIZE)

    # A feasible normalized isotropic start.
    a00.Start = 0.5
    a01.Start = 0.0
    a11.Start = 0.5
    start_W = _paired_normals(np.eye(2) * 0.5)
    start_scores = np.max(r_learn @ start_W.T, axis=1)
    start_q = conformal_quantile(start_scores, alpha)
    q.Start = start_q
    for i in range(n):
        covered[i].Start = float(start_scores[i] <= start_q + 1e-10)
    optimize_t0 = time.perf_counter()
    model.optimize()
    optimizer_time = time.perf_counter() - optimize_t0
    wall_time = time.perf_counter() - total_t0

    status_names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    status = status_names.get(model.Status, f"STATUS_{model.Status}")
    if model.SolCount == 0:
        return GeometryFit(
            learner="Exact MIQCP",
            W=None,
            train_radius=float("nan"),
            wall_time=wall_time,
            status=status,
            optimizer_time=optimizer_time,
            best_bound=float(model.ObjBound) if hasattr(model, "ObjBound") else float("nan"),
            solution_count=0,
            n_variables=int(model.NumVars),
            n_binary=int(model.NumBinVars),
            n_constraints=int(model.NumConstrs + model.NumQConstrs),
        )

    A = np.array([[a00.X, a01.X], [a01.X, a11.X]])
    W = _paired_normals(A)
    incumbent = float(model.ObjVal)
    best_bound = float(model.ObjBound)
    gap_abs = max(0.0, incumbent - best_bound)
    gap_rel = float(model.MIPGap)
    certified = bool(model.Status == GRB.OPTIMAL and gap_rel <= mip_gap + 1e-9)
    return GeometryFit(
        learner="Exact MIQCP",
        W=W,
        train_radius=float(q.X),
        wall_time=wall_time,
        status=status,
        optimizer_time=optimizer_time,
        incumbent=incumbent,
        best_bound=best_bound,
        gap_abs=gap_abs,
        gap_rel=gap_rel,
        certified=certified,
        node_count=float(model.NodeCount),
        work=float(model.Work),
        solution_count=int(model.SolCount),
        n_variables=int(model.NumVars),
        n_binary=int(model.NumBinVars),
        n_constraints=int(model.NumConstrs + model.NumQConstrs),
    )


def _fit_pinball_lbfgsb(
    r_learn: np.ndarray,
    center: np.ndarray,
    seed: int,
    eig_lb: float,
    n_starts: int,
    maxiter: int,
    pinball_weight: float,
) -> GeometryFit:
    """Full-batch smoothed comparator retained for diagnostic use."""
    rng = np.random.default_rng(seed + 711)
    scale = 1.0 / math.sqrt(2.0)
    psi_min = math.asin(eig_lb / scale)
    psi_max = math.pi / 2.0 - psi_min
    max_norm = float(np.max(np.linalg.norm(r_learn, axis=1)))
    max_radius = max(max_norm / math.sqrt(2.0), 0.05)
    tau = min(int(math.ceil((len(r_learn) + 1) * 0.90)), len(r_learn)) / len(r_learn)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, float]:
        A = _paired_spd_matrix(float(x[0]), float(x[1]))
        radius = float(math.exp(float(x[2])))
        return _paired_normals(A), radius

    def surrogate(x: np.ndarray) -> float:
        W, radius = unpack(x)
        smooth_scores = _logsumexp_scores(
            r_learn,
            W,
            np.zeros(4),
            beta=25.0,
        )
        pinball = _smooth_pinball_loss(smooth_scores - radius, tau=tau, beta=25.0)
        robust_obj = _wc_objective_fixed_radius(
            W,
            radius,
            center[None, :],
            DEFAULT_COST,
            LINEAR_LOSS,
        )
        if not np.isfinite(robust_obj):
            return 1e6
        return float(robust_obj + pinball_weight * pinball)

    bounds = [
        (0.0, math.pi),
        (psi_min, psi_max),
        (math.log(1e-3), math.log(max_radius)),
    ]
    best = None
    t0 = time.perf_counter()
    for start in range(n_starts):
        phi0 = 0.0 if start == 0 else float(rng.uniform(0.0, math.pi))
        psi0 = math.pi / 4.0 if start == 0 else float(rng.uniform(psi_min, psi_max))
        W0 = _paired_normals(_paired_spd_matrix(phi0, psi0))
        q0 = conformal_quantile(np.max(r_learn @ W0.T, axis=1), alpha=0.10)
        x0 = np.array([phi0, psi0, math.log(max(q0, 1e-3))])
        result = minimize(
            surrogate,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-10, "maxls": 30},
        )
        candidate = (float(result.fun), result)
        if best is None or candidate[0] < best[0]:
            best = candidate

    wall_time = time.perf_counter() - t0
    assert best is not None
    result = best[1]
    W, radius = unpack(np.asarray(result.x))
    return GeometryFit(
        learner="Pinball L-BFGS-B",
        W=W,
        train_radius=radius,
        wall_time=wall_time,
        status="CONVERGED" if result.success else "ITERATION_LIMIT",
        raw_surrogate_objective=float(result.fun),
    )


def _paired_spd_derivatives(phi: float, psi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A(phi, psi) and its derivatives in the normalized SPD class."""
    scale = 1.0 / math.sqrt(2.0)
    c, s = math.cos(phi), math.sin(phi)
    rotation = np.array([[c, -s], [s, c]])
    d_rotation = np.array([[-s, -c], [c, -s]])
    diagonal = np.diag([scale * math.cos(psi), scale * math.sin(psi)])
    d_diagonal = np.diag([-scale * math.sin(psi), scale * math.cos(psi)])
    A = rotation @ diagonal @ rotation.T
    d_phi = d_rotation @ diagonal @ rotation.T + rotation @ diagonal @ d_rotation.T
    d_psi = rotation @ d_diagonal @ rotation.T
    return A, d_phi, d_psi


def _pinball_surrogate_value(
    x: np.ndarray,
    r_learn: np.ndarray,
    center: np.ndarray,
    tau: float,
    pinball_weight: float,
) -> float:
    """Evaluate the exact nonsmooth surrogate used by Algorithm 2."""
    phi, psi, radius, z0 = (float(item) for item in x)
    A = _paired_spd_matrix(phi, psi)
    scores = np.max(np.abs(r_learn @ A.T), axis=1)
    error = scores - radius
    pinball = np.mean(
        np.where(error >= 0.0, tau * error, -(1.0 - tau) * error)
    )
    z = np.array([z0, 1.0 - z0])
    support = radius * np.sum(np.abs(np.linalg.solve(A, z)))
    base = DEFAULT_COST + np.asarray(center, dtype=float)
    return float(base @ z + support + pinball_weight * pinball)


def _pinball_stochastic_subgradient(
    x: np.ndarray,
    r_batch: np.ndarray,
    center: np.ndarray,
    tau: float,
    pinball_weight: float,
) -> np.ndarray:
    """One minibatch subgradient of the exact pinball-surrogate objective."""
    phi, psi, radius, z0 = (float(item) for item in x)
    A, d_phi, d_psi = _paired_spd_derivatives(phi, psi)
    A_inv = np.linalg.inv(A)
    z = np.array([z0, 1.0 - z0])
    u = A_inv @ z
    sign_u = np.sign(u)

    # Envelope subgradient of r ||A^{-1} z||_1.
    robust_phi = -radius * float(sign_u @ A_inv @ d_phi @ u)
    robust_psi = -radius * float(sign_u @ A_inv @ d_psi @ u)
    robust_radius = float(np.sum(np.abs(u)))
    dz = np.array([1.0, -1.0])
    base = DEFAULT_COST + np.asarray(center, dtype=float)
    robust_z0 = float(base[0] - base[1] + radius * (sign_u @ A_inv @ dz))

    transformed = r_batch @ A.T
    active = np.argmax(np.abs(transformed), axis=1)
    row = np.arange(len(r_batch))
    active_sign = np.sign(transformed[row, active])
    dscore_phi = active_sign * (r_batch @ d_phi.T)[row, active]
    dscore_psi = active_sign * (r_batch @ d_psi.T)[row, active]
    scores = np.abs(transformed[row, active])
    error = scores - radius
    xi = np.where(error > 0.0, tau, np.where(error < 0.0, -(1.0 - tau), 0.0))

    return np.array(
        [
            robust_phi + pinball_weight * float(np.mean(xi * dscore_phi)),
            robust_psi + pinball_weight * float(np.mean(xi * dscore_psi)),
            robust_radius - pinball_weight * float(np.mean(xi)),
            robust_z0,
        ]
    )


def _fit_pinball_stochastic(
    r_learn: np.ndarray,
    center: np.ndarray,
    seed: int,
    eig_lb: float,
    n_starts: int,
    iterations: int,
    batch_size: int,
    step_size: float,
    pinball_weight: float,
) -> GeometryFit:
    """Fixed-context projected minibatch specialization of Algorithm 2.

    The variables are (phi, psi, r, z_1).  Projection enforces the identical
    normalized SPD-paired geometry class and simplex decision used by the
    MIQCP.  The robust support function and nonsmooth max/pinball terms are
    differentiated analytically; no template library is searched.
    """
    r_learn = np.asarray(r_learn, dtype=float)
    rng = np.random.default_rng(seed + 711)
    scale = 1.0 / math.sqrt(2.0)
    psi_min = math.asin(eig_lb / scale)
    psi_max = math.pi / 2.0 - psi_min
    max_norm = float(np.max(np.linalg.norm(r_learn, axis=1)))
    max_radius = max(max_norm / math.sqrt(2.0), 0.05)
    tau = min(int(math.ceil((len(r_learn) + 1) * 0.90)), len(r_learn)) / len(r_learn)
    batch_size = min(max(1, int(batch_size)), len(r_learn))

    best_value = float("inf")
    best_x: np.ndarray | None = None
    t0 = time.perf_counter()
    for start in range(int(n_starts)):
        phi = 0.0 if start == 0 else float(rng.uniform(0.0, math.pi))
        psi = math.pi / 4.0 if start == 0 else float(rng.uniform(psi_min, psi_max))
        W0 = _paired_normals(_paired_spd_matrix(phi, psi))
        radius = conformal_quantile(np.max(r_learn @ W0.T, axis=1), alpha=0.10)

        # Initialize the scalar simplex decision by a deterministic grid search.
        candidate_z = np.linspace(0.0, 1.0, 201)
        candidate_values = []
        A0 = _paired_spd_matrix(phi, psi)
        base = DEFAULT_COST + np.asarray(center, dtype=float)
        for z0 in candidate_z:
            z = np.array([z0, 1.0 - z0])
            candidate_values.append(base @ z + radius * np.sum(np.abs(np.linalg.solve(A0, z))))
        z0 = float(candidate_z[int(np.argmin(candidate_values))])
        x = np.array([phi, psi, radius, z0], dtype=float)

        start_best_x = x.copy()
        start_best_value = _pinball_surrogate_value(
            x, r_learn, center, tau, pinball_weight
        )
        for iteration in range(int(iterations)):
            if batch_size == len(r_learn):
                batch = r_learn
            else:
                indices = rng.choice(len(r_learn), size=batch_size, replace=False)
                batch = r_learn[indices]
            gradient = _pinball_stochastic_subgradient(
                x, batch, center, tau, pinball_weight
            )
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 50.0:
                gradient *= 50.0 / gradient_norm
            eta = float(step_size) / math.sqrt(1.0 + iteration / 100.0)
            x -= eta * gradient
            x[0] %= math.pi
            x[1] = float(np.clip(x[1], psi_min, psi_max))
            x[2] = float(np.clip(x[2], 1e-3, max_radius))
            x[3] = float(np.clip(x[3], 0.0, 1.0))

            if (iteration + 1) % 25 == 0 or iteration + 1 == int(iterations):
                value = _pinball_surrogate_value(
                    x, r_learn, center, tau, pinball_weight
                )
                if value < start_best_value:
                    start_best_value = value
                    start_best_x = x.copy()

        if start_best_value < best_value:
            best_value = start_best_value
            best_x = start_best_x

    wall_time = time.perf_counter() - t0
    assert best_x is not None
    W = _paired_normals(_paired_spd_matrix(float(best_x[0]), float(best_x[1])))
    return GeometryFit(
        learner="Stochastic pinball",
        W=W,
        train_radius=float(best_x[2]),
        wall_time=wall_time,
        status="FIXED_ITERATIONS",
        raw_surrogate_objective=float(best_value),
    )


def _evaluate_geometry(
    fit: GeometryFit,
    data: PreparedData,
    center: np.ndarray,
    max_robust_points: int,
) -> dict:
    base = {
        "learner": fit.learner,
        "status": fit.status,
        "certified": int(fit.certified),
        "solution_count": fit.solution_count,
        "solver_incumbent": fit.incumbent,
        "solver_bound": fit.best_bound,
        "solver_gap_abs": fit.gap_abs,
        "solver_gap_rel": fit.gap_rel,
        "nodes": fit.node_count,
        "work": fit.work,
        "n_variables": fit.n_variables,
        "n_binary": fit.n_binary,
        "n_constraints": fit.n_constraints,
        "fit_time": fit.wall_time,
        "optimizer_time": fit.optimizer_time,
        "surrogate_objective": fit.raw_surrogate_objective,
        "train_radius": fit.train_radius,
    }
    if fit.W is None:
        return {
            **base,
            **{
                key: float("nan")
                for key in [
                    "exact_learn_objective",
                    "exact_learn_radius",
                    "exact_learn_coverage",
                    "recal_radius",
                    "train_radius_learning_coverage",
                    "train_radius_test_coverage",
                    "learn_radius_test_coverage",
                    "post_coverage",
                    "volume",
                    "avg_robust_objective",
                    "avg_realized_regret",
                ]
            },
        }

    exact_obj, learn_radius = _exact_learning_objective(fit.W, data.r_learn, center)
    learn_set = SetResult(
        fit.learner,
        kind="polytope",
        color="#000000",
        W=fit.W,
        h=learn_radius * np.ones(4),
    )
    raw_set = SetResult(
        fit.learner,
        kind="polytope",
        color="#000000",
        W=fit.W,
        h=fit.train_radius * np.ones(4),
    )
    recal_radius = conformal_quantile(np.max(data.r_recal @ fit.W.T, axis=1), alpha=0.10)
    final_set = SetResult(
        fit.learner,
        kind="polytope",
        color="#000000",
        W=fit.W,
        h=recal_radius * np.ones(4),
    )
    metrics = average_robust_metrics(
        final_set,
        centers=data.centers_test,
        y_true=data.y_test,
        cost=DEFAULT_COST,
        loss_spec=LINEAR_LOSS,
        max_points=max_robust_points,
    )
    return {
        **base,
        "exact_learn_objective": exact_obj,
        "exact_learn_radius": learn_radius,
        "exact_learn_coverage": empirical_coverage(learn_set, data.r_learn),
        "recal_radius": recal_radius,
        "train_radius_learning_coverage": empirical_coverage(raw_set, data.r_learn),
        "train_radius_test_coverage": empirical_coverage(raw_set, data.r_test),
        "learn_radius_test_coverage": empirical_coverage(learn_set, data.r_test),
        "post_coverage": empirical_coverage(final_set, data.r_test),
        "volume": set_area(final_set),
        "avg_robust_objective": metrics["avg_robust_objective"],
        "avg_realized_regret": metrics["avg_realized_regret"],
    }


def _coverage_calibration_rows(
    fit: GeometryFit,
    data: PreparedData,
    seed: int,
    n_learn: int,
    targets: list[float],
) -> list[dict]:
    """Evaluate exact independent recalibration over prescribed coverage levels."""
    if fit.W is None:
        return []
    recal_scores = np.max(data.r_recal @ fit.W.T, axis=1)
    test_scores = np.max(data.r_test @ fit.W.T, axis=1)
    output: list[dict] = []
    for target in targets:
        if not 0.0 < target < 1.0:
            raise ValueError("Coverage targets must lie strictly between zero and one.")
        radius = conformal_quantile(recal_scores, alpha=1.0 - target)
        output.append(
            {
                "study": "coverage_calibration",
                "seed": seed,
                "n_learn": n_learn,
                "n_recal": len(data.r_recal),
                "n_test": len(data.r_test),
                "learner": fit.learner,
                "status": fit.status,
                "certified": int(fit.certified),
                "target_coverage": target,
                "recal_radius": radius,
                "empirical_test_coverage": float(
                    np.mean(test_scores <= radius + 1e-12)
                ),
            }
        )
    return output


def _run_exact_study(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    coverage_rows: list[dict] = []
    n_values = _parse_ints(args.exact_n)
    coverage_targets = _parse_floats(args.coverage_targets)
    learning_pool_size = max(max(n_values), int(args.exact_learning_pool))
    seeds = list(range(args.exact_seed, args.exact_seed + args.exact_seeds))
    for n_learn in n_values:
        for seed in seeds:
            print(f"[exact] n={n_learn}, seed={seed}", flush=True)
            data = _prepare_data(
                seed,
                n_learn,
                args.n_recal,
                args.n_test,
                n_learn_pool=learning_pool_size,
            )
            center = data.centers_learn.mean(axis=0)
            exact_fit = _solve_exact_miqcp(
                data.r_learn,
                center,
                seed=seed,
                time_limit=args.time_limit,
                mip_gap=args.mip_gap,
                eig_lb=args.eig_lb,
                threads=args.threads,
            )
            if args.pinball_optimizer == "stochastic":
                pinball_fit = _fit_pinball_stochastic(
                    data.r_learn,
                    center,
                    seed=seed,
                    eig_lb=args.eig_lb,
                    n_starts=args.pinball_starts,
                    iterations=args.pinball_iterations,
                    batch_size=args.pinball_batch_size,
                    step_size=args.pinball_step_size,
                    pinball_weight=args.pinball_weight,
                )
            else:
                pinball_fit = _fit_pinball_lbfgsb(
                    data.r_learn,
                    center,
                    seed=seed,
                    eig_lb=args.eig_lb,
                    n_starts=args.pinball_starts,
                    maxiter=args.pinball_iterations,
                    pinball_weight=args.pinball_weight,
                )
            exact_row = {
                "study": "exact_vs_pinball",
                "seed": seed,
                "n_learn": n_learn,
                "n_recal": args.n_recal,
                "n_test": args.n_test,
                **_evaluate_geometry(exact_fit, data, center, args.max_robust_points),
            }
            pinball_row = {
                "study": "exact_vs_pinball",
                "seed": seed,
                "n_learn": n_learn,
                "n_recal": args.n_recal,
                "n_test": args.n_test,
                **_evaluate_geometry(pinball_fit, data, center, args.max_robust_points),
            }

            pin_obj = float(pinball_row["exact_learn_objective"])
            exact_eval_obj = float(exact_row["exact_learn_objective"])
            bound = float(exact_row["solver_bound"])
            incumbent = float(exact_row["solver_incumbent"])
            pinball_row["certified_excess_upper"] = (
                max(0.0, pin_obj - bound) if np.isfinite(bound) else float("nan")
            )
            pinball_row["incumbent_excess_lower"] = (
                max(0.0, pin_obj - incumbent) if np.isfinite(incumbent) else float("nan")
            )
            pinball_row["paired_objective_difference"] = (
                pin_obj - exact_eval_obj
                if np.isfinite(exact_eval_obj)
                else float("nan")
            )
            pinball_row["paired_relative_difference"] = (
                (pin_obj - exact_eval_obj) / max(abs(exact_eval_obj), 1e-12)
                if np.isfinite(exact_eval_obj)
                else float("nan")
            )
            pinball_row["time_ratio"] = (
                float(exact_row["fit_time"]) / max(float(pinball_row["fit_time"]), 1e-12)
            )
            exact_row["certified_excess_upper"] = float("nan")
            exact_row["incumbent_excess_lower"] = float("nan")
            exact_row["paired_objective_difference"] = float("nan")
            exact_row["paired_relative_difference"] = float("nan")
            exact_row["time_ratio"] = float("nan")
            rows.extend([exact_row, pinball_row])
            coverage_rows.extend(
                _coverage_calibration_rows(
                    exact_fit, data, seed, n_learn, coverage_targets
                )
            )
            coverage_rows.extend(
                _coverage_calibration_rows(
                    pinball_fit, data, seed, n_learn, coverage_targets
                )
            )
            _write_csv(Path(args.exact_output), rows)
            _write_csv(Path(args.coverage_output), coverage_rows)
    return rows, coverage_rows


def _library_factory(kind: str, size: int) -> Callable[[BaselineContext], list[dict]]:
    if kind == "random_cuts":
        return lambda context: _random_cut_templates(context, size)
    if kind == "random_structured":
        return lambda context: _random_templates(context, size)
    if kind == "structured_grid":
        if size % 5 != 0:
            raise ValueError("Structured grid sizes must be a multiple of five.")

        def make_grid(context: BaselineContext) -> list[dict]:
            context.grid_angles = size // 5
            return _template_grid(context)

        return make_grid
    raise ValueError(f"Unknown library kind: {kind}")


def _run_library_study(args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    sizes = _parse_ints(args.library_sizes)
    seeds = list(range(args.library_seed, args.library_seed + args.library_seeds))
    kinds = ["random_cuts", "random_structured", "structured_grid"]
    for seed in seeds:
        data = _prepare_data(seed, 300, args.n_recal, args.n_test)
        context = BaselineContext(
            alpha=0.10,
            y_dim=2,
            cost=DEFAULT_COST,
            robust_center=data.centers_learn.mean(axis=0),
            selection_centers=data.centers_learn[: args.max_selection_centers],
            k_poly=4,
            seed=seed,
            loss_spec=LINEAR_LOSS,
            centers_learn=data.centers_learn,
        )
        for size in sizes:
            for kind in kinds:
                print(f"[library] seed={seed}, kind={kind}, L={size}", flush=True)
                templates = _library_factory(kind, size)(context)
                t0 = time.perf_counter()
                selected = _select_min_robust(templates, data.r_learn, context)
                fit_time = time.perf_counter() - t0
                fitted = _set_from_template(
                    f"{kind}_{size}",
                    "#000000",
                    selected["W"],
                    data.r_recal,
                    context,
                    {
                        "library_kind": kind,
                        "library_size": size,
                        "learn_robust_objective": selected["learn_robust_objective"],
                    },
                )
                metrics = average_robust_metrics(
                    fitted,
                    centers=data.centers_test,
                    y_true=data.y_test,
                    cost=_cost(context),
                    loss_spec=LINEAR_LOSS,
                    max_points=args.max_robust_points,
                )
                rows.append(
                    {
                        "study": "library_sensitivity",
                        "seed": seed,
                        "library_kind": kind,
                        "library_size": size,
                        "n_learn": 300,
                        "n_recal": args.n_recal,
                        "n_test": args.n_test,
                        "learn_objective": selected["learn_robust_objective"],
                        "post_coverage": empirical_coverage(fitted, data.r_test),
                        "volume": set_area(fitted),
                        "avg_robust_objective": metrics["avg_robust_objective"],
                        "avg_realized_regret": metrics["avg_realized_regret"],
                        "selection_time": fit_time,
                    }
                )
                _write_csv(Path(args.library_output), rows)
    return rows


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_std(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


def _aggregate(rows: list[dict], group_keys: list[str], metric_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[item] for item in group_keys)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        summary = dict(zip(group_keys, key))
        summary["runs"] = len(group)
        for metric in metric_keys:
            values = [float(row.get(metric, float("nan"))) for row in group]
            summary[f"{metric}_mean"] = _safe_mean(values)
            summary[f"{metric}_std"] = _safe_std(values)
        if "certified" in group[0]:
            summary["certified_runs"] = int(sum(int(row.get("certified", 0)) for row in group))
        output.append(summary)
    return output


def _paired_exact_rows(rows: list[dict]) -> tuple[dict[tuple[int, int], dict[str, dict]], list[int], str]:
    """Index the matched exact/pinball records."""
    paired: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["n_learn"]), int(row["seed"]))
        paired.setdefault(key, {})[str(row["learner"])] = row
    n_values = sorted({key[0] for key in paired})
    learner_names = sorted({str(row["learner"]) for row in rows})
    pinball_name = next(name for name in learner_names if name != "Exact MIQCP")
    return paired, n_values, pinball_name


def _save_quantile_recovery_plot(
    rows: list[dict], pdf_path: Path, png_path: Path
) -> None:
    """Plot test coverage against prescribed coverage after exact recalibration."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available_n = sorted({int(row["n_learn"]) for row in rows})
    displayed_n = 300 if 300 in available_n else max(available_n)
    rows = [row for row in rows if int(row["n_learn"]) == displayed_n]
    targets = sorted({float(row["target_coverage"]) for row in rows})
    exact_rows = [row for row in rows if row["learner"] == "Exact MIQCP"]
    pinball_rows = [row for row in rows if row["learner"] != "Exact MIQCP"]
    groups = [
        (
            [row for row in exact_rows if bool(int(row["certified"]))],
            "MIQCP certified",
            "#2166ac",
            "s",
            "#2166ac",
        ),
        (
            [row for row in exact_rows if not bool(int(row["certified"]))],
            "MIQCP time limit",
            "#2166ac",
            "s",
            "none",
        ),
        (
            pinball_rows,
            "Stochastic pinball",
            "#b2182b",
            "o",
            "#b2182b",
        ),
    ]
    lower = min(targets) - 0.025
    upper = min(1.0, max(targets) + 0.015)
    identity = np.array([lower, upper])
    fig, ax = plt.subplots(figsize=(4.25, 3.25))
    ax.plot(
        identity,
        identity,
        color="#555555",
        linestyle=":",
        linewidth=1.1,
        label="Reference line",
        zorder=1,
    )
    for group_rows, label, color, marker, facecolor in groups:
        if not group_rows:
            continue
        configurations: dict[tuple[int, int], list[dict]] = {}
        for row in group_rows:
            key = (int(row["n_learn"]), int(row["seed"]))
            configurations.setdefault(key, []).append(row)
        individual_curves: list[list[float]] = []
        for configuration_rows in configurations.values():
            individual_curves.append(
                [
                    float(
                        next(
                            row["empirical_test_coverage"]
                            for row in configuration_rows
                            if math.isclose(
                                float(row["target_coverage"]),
                                target,
                                abs_tol=1e-12,
                            )
                        )
                    )
                    for target in targets
                ]
            )
        curve_array = np.asarray(individual_curves, dtype=float)
        means = np.mean(curve_array, axis=0)
        if len(individual_curves) > 1:
            ax.fill_between(
                targets,
                np.min(curve_array, axis=0),
                np.max(curve_array, axis=0),
                color=color,
                alpha=0.13,
                linewidth=0.0,
                zorder=1.4,
            )
        run_word = "run" if len(configurations) == 1 else "runs"
        ax.plot(
            targets,
            means,
            color=color,
            linestyle="-",
            linewidth=1.65,
            marker=marker,
            markersize=5.0,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=0.8,
            label=f"{label} mean ({len(configurations)} {run_word})",
            zorder=2,
        )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Prescribed coverage $1-\alpha$")
    ax.set_ylabel("Empirical test coverage")
    ax.set_title(
        rf"Coverage after exact re-calibration ($n={displayed_n}$)",
        fontsize=10,
    )
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.text(
        0.97,
        0.035,
        "Shaded: run range\nSolid: group means",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    ax.legend(
        frameon=False,
        fontsize=6.6,
        loc="upper left",
        handlelength=2.0,
    )
    fig.tight_layout(pad=0.8)
    for path in (pdf_path, png_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def _save_performance_speed_plot(
    rows: list[dict], pdf_path: Path, png_path: Path
) -> None:
    """Plot mean paired objective increase and runtime against learning-set size."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    paired, n_values, pinball_name = _paired_exact_rows(rows)
    colors = {"Exact MIQCP": "#2166ac", pinball_name: "#b2182b"}
    markers = {"Exact MIQCP": "s", pinball_name: "o"}
    positions = np.arange(len(n_values), dtype=float)
    performance_means: list[float] = []
    time_means: dict[str, list[float]] = {method: [] for method in colors}

    for n_value in n_values:
        pinball_rows = [
            learners[pinball_name]
            for (n, _), learners in paired.items()
            if n == n_value
        ]
        performance_means.append(
            100.0
            * float(
                np.mean(
                    [
                        float(row["paired_relative_difference"])
                        for row in pinball_rows
                    ]
                )
            )
        )
        for method in colors:
            method_rows = [
                learners[method]
                for (n, _), learners in paired.items()
                if n == n_value
            ]
            time_means[method].append(
                float(np.mean([float(row["fit_time"]) for row in method_rows]))
            )

    fig, performance_ax = plt.subplots(figsize=(5.2, 3.0))
    time_ax = performance_ax.twinx()
    bars = performance_ax.bar(
        positions,
        performance_means,
        width=0.48,
        color="#5a7d36",
        alpha=0.88,
        zorder=2,
    )
    for bar, value in zip(bars, performance_means):
        performance_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#3f3f3f",
        )
    for method in colors:
        time_ax.plot(
            positions,
            time_means[method],
            color=colors[method],
            marker=markers[method],
            markersize=4.8,
            markerfacecolor=colors[method],
            markeredgecolor="white",
            markeredgewidth=0.35,
            linewidth=1.65,
            zorder=3,
        )

    performance_ax.set_xticks(positions, [str(value) for value in n_values])
    performance_ax.set_xlabel(r"Learning-set size $n$")
    performance_ax.set_ylabel("Pinball objective increase (%)")
    time_ax.set_ylabel("Wall time (s)")
    performance_ax.set_ylim(
        0.0, max(performance_means) * 1.22
    )
    time_ax.set_ylim(bottom=0.0)
    performance_ax.grid(axis="y", alpha=0.22, linewidth=0.6, zorder=0)
    performance_ax.set_title(
        "Decision quality and runtime by learning-set size", fontsize=10
    )
    handles = [
        Patch(
            facecolor="#5a7d36",
            alpha=0.88,
            label="Pinball objective increase",
        ),
        Line2D(
            [0],
            [0],
            color=colors["Exact MIQCP"],
            marker=markers["Exact MIQCP"],
            markersize=4.5,
            label="MIQCP time",
        ),
        Line2D(
            [0],
            [0],
            color=colors[pinball_name],
            marker=markers[pinball_name],
            markersize=4.5,
            label="Pinball time",
        ),
    ]
    performance_ax.legend(
        handles=handles,
        frameon=False,
        fontsize=6.7,
        ncol=3,
        loc="upper left",
        columnspacing=0.85,
        handlelength=1.35,
    )
    fig.tight_layout(pad=0.8)
    for path in (pdf_path, png_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["all", "exact", "library"], default="all")
    parser.add_argument("--exact-seed", type=int, default=4101)
    parser.add_argument("--library-seed", type=int, default=2026)
    parser.add_argument("--exact-seeds", type=int, default=5)
    parser.add_argument("--library-seeds", type=int, default=5)
    parser.add_argument("--exact-n", type=str, default="20,40,80,160,300,600")
    parser.add_argument("--exact-learning-pool", type=int, default=600)
    parser.add_argument(
        "--coverage-targets",
        type=str,
        default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.975,0.99",
    )
    parser.add_argument("--library-sizes", type=str, default="10,20,40,60,120")
    parser.add_argument("--n-recal", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--max-selection-centers", type=int, default=120)
    parser.add_argument("--max-robust-points", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--mip-gap", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eig-lb", type=float, default=0.15)
    parser.add_argument(
        "--pinball-optimizer",
        choices=["stochastic", "lbfgsb"],
        default="stochastic",
    )
    parser.add_argument("--pinball-starts", type=int, default=8)
    parser.add_argument("--pinball-iterations", type=int, default=4000)
    parser.add_argument("--pinball-batch-size", type=int, default=64)
    parser.add_argument("--pinball-step-size", type=float, default=0.04)
    parser.add_argument("--pinball-weight", type=float, default=15.0)
    parser.add_argument(
        "--exact-output",
        type=Path,
        default=RESULTS_DIR / "exact_by_seed.csv",
    )
    parser.add_argument(
        "--exact-summary",
        type=Path,
        default=RESULTS_DIR / "exact_summary.csv",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=RESULTS_DIR / "coverage_by_seed.csv",
    )
    parser.add_argument(
        "--exact-quantile-plot",
        type=Path,
        default=RESULTS_DIR / "quantile_recovery.pdf",
    )
    parser.add_argument(
        "--exact-quantile-plot-png",
        type=Path,
        default=RESULTS_DIR / "quantile_recovery.png",
    )
    parser.add_argument(
        "--exact-performance-plot",
        type=Path,
        default=RESULTS_DIR / "performance_speed.pdf",
    )
    parser.add_argument(
        "--exact-performance-plot-png",
        type=Path,
        default=RESULTS_DIR / "performance_speed.png",
    )
    parser.add_argument(
        "--library-output",
        type=Path,
        default=RESULTS_DIR / "library_by_seed.csv",
    )
    parser.add_argument(
        "--library-summary",
        type=Path,
        default=RESULTS_DIR / "library_summary.csv",
    )
    args = parser.parse_args()

    print(
        f"Platform: {platform.platform()}; Python: {platform.python_version()}; "
        f"threads={args.threads}; MIQCP limit={args.time_limit:.0f}s",
        flush=True,
    )

    if args.study in ("all", "exact"):
        exact_rows, coverage_rows = _run_exact_study(args)
        exact_summary = _aggregate(
            exact_rows,
            ["n_learn", "learner"],
            [
                "solver_incumbent",
                "solver_bound",
                "solver_gap_abs",
                "solver_gap_rel",
                "exact_learn_objective",
                "certified_excess_upper",
                "paired_objective_difference",
                "paired_relative_difference",
                "time_ratio",
                "exact_learn_coverage",
                "train_radius_learning_coverage",
                "train_radius_test_coverage",
                "learn_radius_test_coverage",
                "post_coverage",
                "avg_robust_objective",
                "avg_realized_regret",
                "fit_time",
                "optimizer_time",
            ],
        )
        _write_csv(Path(args.exact_summary), exact_summary)
        _save_quantile_recovery_plot(
            coverage_rows,
            Path(args.exact_quantile_plot),
            Path(args.exact_quantile_plot_png),
        )
        _save_performance_speed_plot(
            exact_rows,
            Path(args.exact_performance_plot),
            Path(args.exact_performance_plot_png),
        )

    if args.study in ("all", "library"):
        library_rows = _run_library_study(args)
        library_summary = _aggregate(
            library_rows,
            ["library_kind", "library_size"],
            [
                "learn_objective",
                "post_coverage",
                "volume",
                "avg_robust_objective",
                "avg_realized_regret",
                "selection_time",
            ],
        )
        _write_csv(Path(args.library_summary), library_summary)


if __name__ == "__main__":
    main()
