from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .robust import LossSpec, RobustResult, newsvendor_robust_via_support, solve_robust_decision
from .sets import (
    SetResult,
    ball_as_halfspaces,
    box_halfspaces,
    conformal_quantile,
    directions,
    polytope_vertices,
    set_area,
)


def _import_cvxpy():
    """Import cvxpy with a SciPy sparse compatibility shim.

    Some cvxpy releases still access ``spmatrix.__div__``, which was removed
    from newer SciPy sparse matrix classes under Python 3.  Aliasing it to
    ``__truediv__`` restores the expected API before cvxpy imports its SciPy
    wrapper module.
    """
    import scipy.sparse as sp

    try:
        from scipy.sparse._base import _spbase

        for method_name in ["__div__", "__mul__", "__add__", "__sub__", "__le__", "__eq__", "__lt__", "__gt__"]:
            spmatrix_method = getattr(sp.spmatrix, method_name, None)
            base_method = getattr(_spbase, method_name, None)
            if base_method is None:
                continue
            if spmatrix_method is None or getattr(spmatrix_method, "__objclass__", None) is object:
                setattr(sp.spmatrix, method_name, base_method)
    except Exception:
        if not hasattr(sp.spmatrix, "__div__"):
            sp.spmatrix.__div__ = lambda self, other: self / other

    import cvxpy as cp

    return cp


@dataclass
class BaselineContext:
    alpha: float
    y_dim: int = 2
    cost: np.ndarray | None = None
    robust_center: np.ndarray | None = None
    selection_centers: np.ndarray | None = None
    k_poly: int = 4
    n_croms: int = 10
    seed: int = 2026
    grid_angles: int = 24
    grid_rhos: tuple[float, ...] = (0.55, 0.75, 1.0, 1.35, 1.8)
    loss_spec: LossSpec | None = None
    # Full (untruncated) f̂(x_learn) predictions for the learning split.
    # Used by feature-conditional methods (Yeh conditional, PCP generative).
    centers_learn: np.ndarray | None = None
    centers_recal: np.ndarray | None = None
    # Wall-clock limit (seconds) for each exact MIQCP solve used by ``Ours``.
    miqcp_time_limit: float = 1800.0


def _cost(context: BaselineContext) -> np.ndarray:
    if context.cost is None:
        return np.zeros(context.y_dim)
    return np.asarray(context.cost, dtype=float)


def _center(context: BaselineContext) -> np.ndarray:
    if context.robust_center is None:
        return np.zeros(context.y_dim)
    return np.asarray(context.robust_center, dtype=float)


def _selection_centers(context: BaselineContext) -> np.ndarray:
    if context.selection_centers is None:
        return _center(context)[None, :]
    return np.asarray(context.selection_centers, dtype=float)


def _loss(context: BaselineContext) -> LossSpec:
    return context.loss_spec or LossSpec()


def _ensure_2d(context: BaselineContext) -> None:
    if context.y_dim != 2:
        raise ValueError("The current baseline library is for the planned 2D Y experiment.")


def stretched_direction_normals(k: int, phi: float, rho: float) -> np.ndarray:
    if k < 4:
        raise ValueError("Need at least four halfspaces for a bounded 2D template.")
    rot = np.array(
        [
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ]
    )
    transform = np.diag([1.0 / rho, rho]) @ rot.T
    return directions(k) @ transform


def calibrate_template(W: np.ndarray, residuals: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    scores = np.max(residuals @ W.T, axis=1)
    q = conformal_quantile(scores, alpha)
    return q * np.ones(W.shape[0]), q


def _set_from_template(
    name: str,
    color: str,
    W: np.ndarray,
    residuals: np.ndarray,
    context: BaselineContext,
    metadata: dict,
) -> SetResult:
    h, q = calibrate_template(W, residuals, context.alpha)
    full_metadata = {"q": q, "K": W.shape[0], **metadata}
    return SetResult(name, kind="polytope", color=color, W=W, h=h, metadata=full_metadata)


def _template_grid(context: BaselineContext) -> list[dict]:
    _ensure_2d(context)
    phis = np.linspace(0.0, np.pi, context.grid_angles, endpoint=False)
    templates = []
    for phi in phis:
        for rho in context.grid_rhos:
            templates.append(
                {
                    "W": stretched_direction_normals(context.k_poly, float(phi), float(rho)),
                    "phi": float(phi),
                    "rho": float(rho),
                }
            )
    return templates


def _random_templates(context: BaselineContext, n_templates: int | None = None) -> list[dict]:
    _ensure_2d(context)
    count = context.n_croms if n_templates is None else int(n_templates)
    rng = np.random.default_rng(context.seed + 17)
    templates = []
    for _ in range(count):
        phi = float(rng.uniform(0.0, np.pi))
        rho = float(np.exp(rng.uniform(np.log(0.55), np.log(1.8))))
        templates.append(
            {
                "W": stretched_direction_normals(context.k_poly, phi, rho),
                "phi": phi,
                "rho": rho,
            }
        )
    return templates


def _random_cut_templates(context: BaselineContext, n_templates: int | None = None) -> list[dict]:
    """Generate random K-cut polyhedral templates.

    F-CROMS should select from a finite randomly generated library.  Each
    candidate here is an intersection of K halfspaces with randomly oriented
    unit normals; we reject draws whose normals do not positively span R^2,
    because otherwise the calibrated set can be unbounded.
    """
    _ensure_2d(context)
    count = context.n_croms if n_templates is None else int(n_templates)
    rng = np.random.default_rng(context.seed + 17)
    templates = []
    attempts = 0
    while len(templates) < count and attempts < 5000:
        attempts += 1
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, context.k_poly))
        gaps = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
        if float(np.max(gaps)) >= np.pi - 1e-8:
            continue
        W = np.column_stack([np.cos(angles), np.sin(angles)])
        templates.append({"W": W, "angles": angles.tolist()})
    if len(templates) < count:
        raise RuntimeError("Could not generate enough bounded random K-cut templates.")
    return templates


def _paired_random_cut_templates(context: BaselineContext, n_templates: int | None = None) -> list[dict]:
    """Generate bounded K-cut templates in arbitrary dimension.

    Normals are generated in plus/minus pairs, which guarantees positive
    spanning.  This is used for higher-dimensional hyperparameter sweeps where
    the 2D angle/rho grid is not available.
    """
    count = context.n_croms if n_templates is None else int(n_templates)
    d = context.y_dim
    if context.k_poly < 2 * d:
        raise ValueError(f"Need K >= 2d for bounded paired random templates; got K={context.k_poly}, d={d}.")
    rng = np.random.default_rng(context.seed + 901)
    templates = []
    for _ in range(count):
        n_pairs = context.k_poly // 2
        dirs = rng.normal(size=(n_pairs, d))
        dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-10)
        W = np.vstack([dirs, -dirs])
        if W.shape[0] < context.k_poly:
            extra = rng.normal(size=(1, d))
            extra = extra / max(float(np.linalg.norm(extra)), 1e-10)
            W = np.vstack([W, extra])
        scales = np.exp(rng.uniform(np.log(0.65), np.log(1.55), size=W.shape[0]))
        W = W * scales[:, None]
        templates.append({"W": W, "kind": "paired-random"})
    return templates


def _select_min_area(templates: list[dict], residuals: np.ndarray, context: BaselineContext) -> dict:
    best = None
    for template in templates:
        h, q = calibrate_template(template["W"], residuals, context.alpha)
        result = SetResult("candidate", kind="polytope", color="#000000", W=template["W"], h=h)
        metric = set_area(result)
        candidate = {**template, "learn_q": q, "learn_area": metric}
        if best is None or metric < best["learn_area"]:
            best = candidate
    if best is None:
        raise RuntimeError("No templates were available for size-aware selection.")
    return best


def _select_min_robust(templates: list[dict], residuals: np.ndarray, context: BaselineContext) -> dict:
    best = None
    centers = _selection_centers(context)
    cost = _cost(context)
    for template in templates:
        h, q = calibrate_template(template["W"], residuals, context.alpha)
        result = SetResult("candidate", kind="polytope", color="#000000", W=template["W"], h=h)
        objectives = [
            solve_robust_decision(result, center=center, cost=cost, loss_spec=_loss(context)).objective
            for center in centers
        ]
        metric = float(np.mean(objectives))
        candidate = {**template, "learn_q": q, "learn_robust_objective": metric}
        if best is None or metric < best["learn_robust_objective"]:
            best = candidate
    if best is None:
        raise RuntimeError("No templates were available for decision-aware selection.")
    return best


# Exact newsvendor robust decision over a convex set (see robust.py).
_newsvendor_robust_via_support = newsvendor_robust_via_support


class BaseMethod:
    name: str
    supported_experiments: tuple[str, ...] | None = None
    # False for methods whose fit() is a no-op (box, ball, Bonferroni).
    # The runner uses this to give those methods r_learn ∪ r_recal for
    # calibration, so the n_learn sweep is fair: baselines that don't spend
    # any data on shape learning are not penalized by a shrinking n_recal.
    uses_learning_split: bool = True
    calibration_space: str = "residual"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        pass

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        raise NotImplementedError


class BonferroniBox(BaseMethod):
    name = "Bonferroni box"
    uses_learning_split = False
    calibration_space = "residual"

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        per_coord_alpha = context.alpha / context.y_dim
        q = np.array(
            [conformal_quantile(np.abs(r_recal[:, j]), per_coord_alpha) for j in range(context.y_dim)]
        )
        W = np.vstack([np.eye(context.y_dim), -np.eye(context.y_dim)])
        h = np.concatenate([q, q])
        return SetResult(self.name, kind="polytope", color="#8c8c8c", W=W, h=h, metadata={"q": q})


class ConformalBox(BaseMethod):
    name = "Conformal box"
    uses_learning_split = False

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        q = conformal_quantile(np.max(np.abs(r_recal), axis=1), context.alpha)
        W, h = box_halfspaces(q, dim=context.y_dim)
        return SetResult(self.name, kind="polytope", color="#595959", W=W, h=h, metadata={"q": q})


class ConformalBall(BaseMethod):
    name = "Conformal ball"
    uses_learning_split = False

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        q = conformal_quantile(np.linalg.norm(r_recal, axis=1), context.alpha)
        if context.y_dim == 2:
            W, h = ball_as_halfspaces(q)
        else:
            W, h = None, None
        return SetResult(self.name, kind="ball", color="#1f77b4", W=W, h=h, radius=q, metadata={"q": q, "dim": context.y_dim})


class MahalanobisEllipseAdapter(BaseMethod):
    """Conformal ellipse with a Mahalanobis score.

    Score s(r) = ||L^{-1} r||_2 where L = chol(Cov(r_learn)).
    Prediction set {r : s(r) ≤ q} is an ellipse L·Ball(q).
    Robust solver uses the support function h_C(z) = q·||L^T z||_2 (exact, no polygon approx).
    """

    name = "Conformal ellipse (Mahalanobis)"
    color = "#1f77b4"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        cov = np.cov(r_learn.T) + 1e-8 * np.eye(2)
        self.L_ = np.linalg.cholesky(cov)
        self.L_inv_ = np.linalg.inv(self.L_)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        scores = np.linalg.norm(r_recal @ self.L_inv_.T, axis=1)
        q = float(conformal_quantile(scores, context.alpha))
        L = self.L_.copy()
        L_inv = self.L_inv_.copy()
        Sigma = L @ L.T

        def score_fn(r: np.ndarray) -> np.ndarray:
            return np.linalg.norm(np.asarray(r) @ L_inv.T, axis=1 if np.asarray(r).ndim == 2 else None)

        def _golden_section(f: callable, lo: float, hi: float, n_iter: int = 100) -> float:
            inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
            c = hi - inv_phi * (hi - lo)
            d = lo + inv_phi * (hi - lo)
            fc, fd = f(c), f(d)
            for _ in range(n_iter):
                if fc <= fd:
                    hi, d, fd = d, c, fc
                    c = hi - inv_phi * (hi - lo)
                    fc = f(c)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + inv_phi * (hi - lo)
                    fd = f(d)
            return 0.5 * (lo + hi)

        def robust_solver_fn(center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
            center = np.asarray(center, dtype=float)
            cost = np.asarray(cost, dtype=float)
            if loss_spec.name == "newsvendor":
                def support(a: np.ndarray) -> float:
                    return q * float(np.linalg.norm(L.T @ a))

                def support_argmax(a: np.ndarray) -> np.ndarray:
                    nrm = float(np.linalg.norm(L.T @ a))
                    return q * (Sigma @ a) / nrm if nrm > 1e-12 else np.zeros(2)

                return _newsvendor_robust_via_support(support, support_argmax, center, cost, loss_spec)
            base = center + cost

            def obj(t: float) -> float:
                z = np.array([t, 1.0 - t])
                return float(base @ z) + q * float(np.linalg.norm(L.T @ z)) + loss_spec.penalty(z)

            t_opt = _golden_section(obj, 0.0, 1.0)
            z = np.array([t_opt, 1.0 - t_opt])
            Lt_z = L.T @ z
            norm_Lt_z = float(np.linalg.norm(Lt_z))
            if norm_Lt_z > 1e-12:
                worst_r = q * (Sigma @ z) / norm_Lt_z
            else:
                worst_r = np.zeros(2)
            worst_y = center + worst_r
            return RobustResult(
                z=z,
                worst_y=worst_y,
                objective=loss_spec.loss(worst_y, cost, z),
                worst_residual=worst_r,
            )

        return SetResult(
            self.name,
            kind="ellipse",
            color=self.color,
            score_fn=score_fn,
            threshold=q,
            chol_factor=L,
            robust_solver_fn=robust_solver_fn,
            metadata={"q": q},
        )


class FICNNAdapter(BaseMethod):
    name = "FICNN"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        raise NotImplementedError("Implement a small convex-score PyTorch model.")

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        raise NotImplementedError


class _YehPICNNModel:
    """Small copy of Yeh et al.'s PICNN score architecture for 2D residual sets.

    The official repo uses a PICNN s(x, y) whose sublevel sets are convex in y.
    Our experiment pipeline currently uses feature-independent residual sets, so
    the adapter feeds a constant x and learns a convex score in residual space.
    """

    def __init__(
        self,
        y_dim: int,
        hidden_dim: int,
        n_layers: int,
        epsilon: float,
        seed: int,
    ) -> None:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch = torch
        self.nn = nn
        self.input_dim = 1
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.L = n_layers
        self.epsilon = float(epsilon)

        class _Module(nn.Module):
            def __init__(module_self) -> None:
                super().__init__()
                module_self.input_dim = 1
                module_self.y_dim = y_dim
                module_self.hidden_dim = hidden_dim
                module_self.L = n_layers
                module_self.epsilon = float(epsilon)
                module_self.y_in_output_layer = True
                null_module = nn.Module()
                L = n_layers
                d = hidden_dim
                module_self.W_hat_layers = nn.ModuleList(
                    [null_module] + [nn.Linear(d, d) for _ in range(1, L + 1)]
                )
                module_self.W_bar_layers = nn.ModuleList(
                    [null_module]
                    + [nn.Linear(d, d, bias=False) for _ in range(1, L)]
                    + [nn.Linear(d, 1, bias=False)]
                )
                module_self.V_hat_layers = nn.ModuleList(
                    [nn.Linear(1, y_dim)] + [nn.Linear(d, y_dim) for _ in range(1, L + 1)]
                )
                module_self.V_bar_layers = nn.ModuleList(
                    [nn.Linear(y_dim, d, bias=False) for _ in range(L)]
                    + [nn.Linear(y_dim, 1, bias=False)]
                )
                module_self.b_layers = nn.ModuleList(
                    [nn.Linear(1, d)]
                    + [nn.Linear(d, d) for _ in range(1, L)]
                    + [nn.Linear(d, 1)]
                )
                module_self.u_layers = nn.ModuleList(
                    [nn.Linear(1, d)] + [nn.Linear(d, d) for _ in range(1, L)]
                )
                module_self.clamp_weights()

            def clamp_weights(module_self) -> None:
                with torch.no_grad():
                    for layer in module_self.W_bar_layers:
                        if isinstance(layer, nn.Linear):
                            layer.weight.clamp_(min=0.0)

            def forward(module_self, x, y):
                relu = nn.ReLU()
                u = x
                sigma: Any = 0.0
                for layer_idx in range(n_layers):
                    if layer_idx > 0:
                        w_hat = relu(module_self.W_hat_layers[layer_idx](u))
                        w_sigma = module_self.W_bar_layers[layer_idx](w_hat * sigma)
                    else:
                        w_sigma = 0.0
                    v_hat = module_self.V_hat_layers[layer_idx](u)
                    v_y = module_self.V_bar_layers[layer_idx](v_hat * y)
                    b = module_self.b_layers[layer_idx](u)
                    sigma = relu(w_sigma + v_y + b)
                    u = relu(module_self.u_layers[layer_idx](u))

                layer_idx = n_layers
                w_hat = relu(module_self.W_hat_layers[layer_idx](u))
                output = module_self.W_bar_layers[layer_idx](w_hat * sigma)
                v_hat = module_self.V_hat_layers[layer_idx](u)
                output = output + module_self.V_bar_layers[layer_idx](v_hat * y)
                output = output + module_self.b_layers[layer_idx](u)
                if epsilon > 0.0:
                    output = output + epsilon * torch.linalg.norm(y, ord=float("inf"), dim=1, keepdim=True)
                return output[:, 0]

        self.module = _Module()

    def to(self, device: str) -> "_YehPICNNModel":
        self.module.to(device)
        return self

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def clamp_weights(self) -> None:
        self.module.clamp_weights()

    def scores(self, residuals: np.ndarray, batch_size: int = 512) -> np.ndarray:
        torch = self.torch
        self.eval()
        residuals = np.asarray(residuals, dtype=np.float32)
        out = []
        with torch.no_grad():
            for start in range(0, len(residuals), batch_size):
                y = torch.from_numpy(residuals[start : start + batch_size])
                x = torch.zeros((len(y), 1), dtype=torch.float32)
                out.append(self.module(x, y).detach().cpu().numpy())
        return np.concatenate(out)


def _mala_picnn(
    module,
    num_samples: int,
    burnin: int,
    x_init,
    y_init,
    std: float,
):
    """Yeh et al.'s MALA sampler for PICNN/FALCON training."""
    torch = __import__("torch")

    module.eval()
    x_dim = x_init.shape[1]
    xy_init = torch.cat([x_init, y_init], dim=1)

    x_init.requires_grad_(True)
    y_init.requires_grad_(True)
    output_init = module(x_init, y_init)
    init_grad_x, init_grad_y = torch.autograd.grad(
        outputs=output_init,
        inputs=(x_init, y_init),
        grad_outputs=torch.ones_like(output_init),
    )
    mu = xy_init - 0.5 * std**2 * torch.cat([init_grad_x, init_grad_y], dim=1)

    samples = []
    for step in range(burnin + num_samples):
        with torch.no_grad():
            sample_candidate = mu + torch.randn_like(xy_init) * std
        sample_candidate.requires_grad_(True)

        sample_x = sample_candidate[:, :x_dim]
        sample_y = sample_candidate[:, x_dim:]
        output = module(sample_x, sample_y)
        grad_x, grad_y = torch.autograd.grad(
            outputs=output,
            inputs=(sample_x, sample_y),
            grad_outputs=torch.ones_like(output),
        )
        if grad_x is None or grad_y is None:
            raise RuntimeError("PICNN MALA gradient was None.")

        with torch.no_grad():
            nu = sample_candidate - 0.5 * std**2 * torch.cat([grad_x, grad_y], dim=1)
            acceptance = torch.exp(
                -output
                - torch.norm(xy_init - nu, dim=1) ** 2 / (2 * std**2)
                + output_init
                + torch.norm(sample_candidate - mu, dim=1) ** 2 / (2 * std**2)
            )
            acceptance.clamp_(0.0, 1.0)
            accept = torch.bernoulli(acceptance).to(dtype=torch.bool)

            mu[accept] = nu[accept]
            xy_init[accept] = sample_candidate[accept]
            output_init[accept] = output[accept]
            if step >= burnin:
                samples.append(xy_init.clone())

    return torch.stack(samples)


def _fit_yeh_picnn_falcon(
    residuals: np.ndarray,
    context: BaselineContext,
    hidden_dim: int = 64,
    n_layers: int = 2,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-2,
    l2reg: float = 1e-4,
    num_chains: int = 1,
    num_samples: int = 1,
    burnin: int = 100,
    std: float = 0.25,
    warm_start: bool = False,
    weight_towards_zero: float = 1.0,
    epsilon: float = 1.0,
) -> _YehPICNNModel:
    """Fit the PICNN score with Yeh et al.'s FALCON/MALA objective.

    The public Yeh implementation fits a log-concave PICNN energy by minimizing
    ``E_data[s(x,y)] - E_MALA[s(x,y)]`` with optional shrinkage of data scores
    toward zero.  Our residual-set experiment has no conditioning features in
    the baseline API, so this adapter uses the minimal constant-x reduction.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    residuals = np.asarray(residuals, dtype=np.float32)
    torch.manual_seed(context.seed + 409)

    model = _YehPICNNModel(
        y_dim=context.y_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epsilon=epsilon,
        seed=context.seed + 409,
    )

    device = "cpu"
    module = model.module.eval().to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=l2reg)
    x_data = torch.zeros((len(residuals), model.input_dim), dtype=torch.float32)
    y_data = torch.from_numpy(residuals)
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=True)

    x_init = torch.zeros((num_chains, model.input_dim), device=device)
    y_init = torch.zeros((num_chains, model.y_dim), device=device)
    train_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        n_seen = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            samples = _mala_picnn(
                module,
                num_samples=num_samples,
                burnin=burnin,
                x_init=x_init,
                y_init=y_init,
                std=std,
            ).detach()
            optimizer.zero_grad()

            data_out = module(x_batch, y_batch)
            samples_out = module(
                samples[:, :, : model.input_dim].reshape(-1, model.input_dim),
                samples[:, :, model.input_dim :].reshape(-1, model.y_dim),
            )
            loss = data_out.mean() - samples_out.mean()
            if weight_towards_zero > 0.0:
                loss = loss + weight_towards_zero * (data_out**2).mean()
            loss.backward()
            optimizer.step()
            model.clamp_weights()

            total_loss += float(loss.item()) * len(x_batch)
            n_seen += len(x_batch)
            if warm_start:
                indices = torch.randint(0, num_chains, (num_chains,))
                x_init = samples[-1, indices, : model.input_dim]
                y_init = samples[-1, indices, model.input_dim :]

        train_losses.append(total_loss / max(1, n_seen))
    model.train_losses_ = train_losses
    return model


class _YehPICNNModelConditional:
    """PICNN with a variable-dimension x input (for conditioning on ŷ).

    Identical architecture to _YehPICNNModel but the x-pathway layers are sized
    to x_dim instead of 1, so the score s(x, r) is convex in r for every x.
    """

    def __init__(self, y_dim: int, x_dim: int, hidden_dim: int, n_layers: int,
                 epsilon: float, seed: int, y_in_output_layer: bool = False) -> None:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        self.torch = torch
        self.nn = nn
        self.input_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.L = n_layers
        self.epsilon = float(epsilon)
        self.y_in_output_layer = bool(y_in_output_layer)

        class _Module(nn.Module):
            def __init__(ms) -> None:
                super().__init__()
                ms.input_dim = x_dim
                ms.y_dim = y_dim
                ms.hidden_dim = hidden_dim
                ms.L = n_layers
                ms.epsilon = float(epsilon)
                ms.y_in_output_layer = bool(y_in_output_layer)
                null_mod = nn.Module()
                L, d = n_layers, hidden_dim
                ms.W_hat_layers = nn.ModuleList(
                    [null_mod] + [nn.Linear(d, d) for _ in range(1, L + 1)])
                ms.W_bar_layers = nn.ModuleList(
                    [null_mod] + [nn.Linear(d, d, bias=False) for _ in range(1, L)]
                    + [nn.Linear(d, 1, bias=False)])
                ms.V_hat_layers = nn.ModuleList(
                    [nn.Linear(x_dim, y_dim)] + [nn.Linear(d, y_dim) for _ in range(1, L + 1)])
                ms.V_bar_layers = nn.ModuleList(
                    [nn.Linear(y_dim, d, bias=False) for _ in range(L)]
                    + [nn.Linear(y_dim, 1, bias=False)])
                ms.b_layers = nn.ModuleList(
                    [nn.Linear(x_dim, d)] + [nn.Linear(d, d) for _ in range(1, L)]
                    + [nn.Linear(d, 1)])
                ms.u_layers = nn.ModuleList(
                    [nn.Linear(x_dim, d)] + [nn.Linear(d, d) for _ in range(1, L)])
                ms.clamp_weights()

            def clamp_weights(ms) -> None:
                with torch.no_grad():
                    for layer in ms.W_bar_layers:
                        if isinstance(layer, nn.Linear):
                            layer.weight.clamp_(min=0.0)
                    if not ms.y_in_output_layer:
                        ms.V_bar_layers[-1].weight.fill_(0.0)

            def forward(ms, x, y):
                from typing import Any
                relu = nn.ReLU()
                u = x
                sigma: Any = 0.0
                for idx in range(n_layers):
                    if idx > 0:
                        w_hat = relu(ms.W_hat_layers[idx](u))
                        w_sigma = ms.W_bar_layers[idx](w_hat * sigma)
                    else:
                        w_sigma = 0.0
                    v_hat = ms.V_hat_layers[idx](u)
                    v_y = ms.V_bar_layers[idx](v_hat * y)
                    b = ms.b_layers[idx](u)
                    sigma = relu(w_sigma + v_y + b)
                    u = relu(ms.u_layers[idx](u))
                idx = n_layers
                w_hat = relu(ms.W_hat_layers[idx](u))
                output = ms.W_bar_layers[idx](w_hat * sigma)
                if ms.y_in_output_layer:
                    v_hat = ms.V_hat_layers[idx](u)
                    output = output + ms.V_bar_layers[idx](v_hat * y)
                output = output + ms.b_layers[idx](u)
                if ms.epsilon != 0.0:
                    output = output + ms.epsilon * torch.linalg.norm(
                        y, ord=float("inf"), dim=1, keepdim=True)
                return output[:, 0]

        self.module = _Module()

    def to(self, device: str) -> "_YehPICNNModelConditional":
        self.module.to(device)
        return self

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def clamp_weights(self) -> None:
        self.module.clamp_weights()

    def scores_conditional(self, x: np.ndarray, residuals: np.ndarray,
                           batch_size: int = 512) -> np.ndarray:
        torch = self.torch
        self.eval()
        x = np.asarray(x, dtype=np.float32)
        residuals = np.asarray(residuals, dtype=np.float32)
        if len(residuals) == 0:
            return np.array([], dtype=np.float32)
        out = []
        with torch.no_grad():
            for s in range(0, len(residuals), batch_size):
                xb = torch.from_numpy(x[s: s + batch_size])
                rb = torch.from_numpy(residuals[s: s + batch_size])
                out.append(self.module(xb, rb).detach().cpu().numpy())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)


def _fit_yeh_picnn_falcon_conditional(
    x_learn: np.ndarray,
    r_learn: np.ndarray,
    context: BaselineContext,
    hidden_dim: int = 128,
    n_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-2,
    l2reg: float = 1e-3,
    num_chains: int = 50,
    num_samples: int = 1,
    burnin: int = 100,
    std: float = 0.25,
    warm_start: bool = False,
    weight_towards_zero: float = 1.0,
    epsilon: float = 0.0,
    y_in_output_layer: bool = False,
) -> _YehPICNNModelConditional:
    """Fit conditional PICNN s(x, r) with Yeh et al.'s FALCON/MALA objective.

    Defaults match ``run_portfolio_picnn.py`` in chrisyeh96/e2e-conformal (ETO).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_learn = np.asarray(x_learn, dtype=np.float32)
    r_learn = np.asarray(r_learn, dtype=np.float32)
    x_dim = x_learn.shape[1]
    torch.manual_seed(context.seed + 711)

    model = _YehPICNNModelConditional(
        y_dim=context.y_dim,
        x_dim=x_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epsilon=epsilon,
        seed=context.seed + 711,
        y_in_output_layer=y_in_output_layer,
    )
    device = "cpu"
    module = model.module.eval().to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=l2reg)

    x_data = torch.from_numpy(x_learn)
    y_data = torch.from_numpy(r_learn)
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=batch_size, shuffle=True)

    x_init = torch.zeros((num_chains, x_dim), device=device)
    y_init = torch.zeros((num_chains, model.y_dim), device=device)

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            samples = _mala_picnn(
                module, num_samples, burnin, x_init, y_init, std
            ).detach()
            optimizer.zero_grad()

            data_out = module(x_batch, y_batch)
            samples_out = module(
                samples[:, :, :x_dim].reshape(-1, x_dim),
                samples[:, :, x_dim:].reshape(-1, model.y_dim),
            )
            loss = data_out.mean() - samples_out.mean()
            if weight_towards_zero > 0.0:
                loss = loss + weight_towards_zero * (data_out**2).mean()
            loss.backward()
            optimizer.step()
            model.clamp_weights()

            if warm_start:
                indices = torch.randint(0, num_chains, (num_chains,))
                x_init = samples[-1, indices, :x_dim]
                y_init = samples[-1, indices, x_dim:]
    return model


def _convex_polygon_from_picnn_conditional(
    model: _YehPICNNModelConditional,
    threshold: float,
    x_input: np.ndarray,
    n_dirs: int = 64,
    cal_residuals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate {r : s(x_input, r) <= threshold} as a polygon.

    x_input: shape (1, x_dim) — the conditioning ŷ value.
    """
    import torch

    x_input = np.asarray(x_input, dtype=np.float32)
    x_vec = np.asarray(x_input, dtype=float).reshape(-1)
    threshold, center = _picnn_q_feasible_conditional(model, x_vec, threshold)

    def _score(r: np.ndarray) -> float:
        return float(model.scores_conditional(
            np.tile(x_input, (len(r), 1)) if r.ndim == 2 else x_input,
            r[None] if r.ndim == 1 else r,
        )[0 if r.ndim == 1 else slice(None)])

    def _scores_batch(r: np.ndarray) -> np.ndarray:
        x_tiled = np.tile(x_input, (len(r), 1))
        return model.scores_conditional(x_tiled, r.astype(np.float32))

    # center from min-score problem is inside the feasible sublevel set at q_feas
    if _scores_batch(center[None])[0] > threshold + 1e-6:
        y_t = torch.zeros((1, model.y_dim), dtype=torch.float32, requires_grad=True)
        x_t = torch.from_numpy(x_input)
        opt = torch.optim.Adam([y_t], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            model.module(x_t, y_t).mean().backward()
            opt.step()
        center = y_t.detach().cpu().numpy()[0].astype(float)
        if _scores_batch(center[None])[0] > threshold:
            center = np.zeros(model.y_dim, dtype=float)

    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    if cal_residuals is not None and len(cal_residuals):
        cal_residuals = np.asarray(cal_residuals, dtype=float)
        base_radius = float(np.max(np.linalg.norm(cal_residuals - center, axis=1)) + 1.0)
    else:
        base_radius = max(1.0, float(np.linalg.norm(center)) + 1.0)
    max_radius = max(100.0, 64.0 * base_radius)
    points = []
    for direction in dirs:
        lo, hi = 0.0, max(base_radius, 1.0)
        found_boundary = False
        for _ in range(30):
            if _scores_batch((center + hi * direction)[None])[0] > threshold:
                found_boundary = True
                break
            hi *= 2.0
            if hi > max_radius:
                break
        if not found_boundary:
            points.append(center + max_radius * direction)
            continue
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _scores_batch((center + mid * direction)[None])[0] <= threshold:
                lo = mid
            else:
                hi = mid
        points.append(center + lo * direction)
    verts = np.vstack(points)
    W, h = [], []
    for i in range(len(verts)):
        p, q_v = verts[i], verts[(i + 1) % len(verts)]
        edge = q_v - p
        normal = np.array([edge[1], -edge[0]])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        W.append(normal)
        h.append(float(normal @ p))
    if not W:
        radius = float(np.max(np.linalg.norm(verts - center, axis=1)))
        radius = max(radius, 1e-4)
        W, h = ball_as_halfspaces(radius)
    else:
        W = np.vstack(W)
        h = np.asarray(h)
    return W, h, verts, center


def _picnn_lifted_parameters_conditional(
    model: _YehPICNNModelConditional, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Lifted PICNN parameters at a fixed conditioning input x (Yeh et al.)."""
    torch = model.torch
    relu = model.nn.ReLU()
    module = model.module
    L = model.L
    d = model.hidden_dim
    x_vec = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(-1))

    V_stacked = np.zeros((L * d + 1, model.y_dim), dtype=float)
    Vs = [V_stacked[d * layer : d * (layer + 1)] for layer in range(L + 1)]
    W_stacked = np.zeros(((L - 1) * d + 1, d), dtype=float)
    Ws = [W_stacked[d * layer : d * (layer + 1)] for layer in range(L)]
    bs = np.zeros((L, d), dtype=float)

    with torch.no_grad():
        u = x_vec.to(torch.float32)
        for layer in range(L):
            if layer > 0:
                w_hat = relu(module.W_hat_layers[layer](u))
                W = module.W_bar_layers[layer].weight @ torch.diag(w_hat)
                Ws[layer - 1][:] = W.detach().cpu().numpy()
            v_hat = module.V_hat_layers[layer](u)
            V = module.V_bar_layers[layer].weight @ torch.diag(v_hat)
            b = module.b_layers[layer](u)
            Vs[layer][:] = V.detach().cpu().numpy()
            bs[layer] = b.detach().cpu().numpy()
            u = relu(module.u_layers[layer](u))

        layer = L
        w_hat = relu(module.W_hat_layers[layer](u))
        W = module.W_bar_layers[layer].weight @ torch.diag(w_hat)
        Ws[layer - 1][:] = W.detach().cpu().numpy()
        v_hat = module.V_hat_layers[layer](u)
        V = module.V_bar_layers[layer].weight @ torch.diag(v_hat)
        Vs[-1][:] = V.detach().cpu().numpy()
        b_fin = float(module.b_layers[layer](u)[0].detach().cpu().numpy())

    return V_stacked, W_stacked, bs, b_fin


def _picnn_lifted_parameters(model: _YehPICNNModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Populate Yeh et al.'s lifted PICNN optimization parameters at x=0."""
    torch = model.torch
    relu = model.nn.ReLU()
    module = model.module
    L = model.L
    d = model.hidden_dim

    V_stacked = np.zeros((L * d + 1, model.y_dim), dtype=float)
    Vs = [V_stacked[d * layer : d * (layer + 1)] for layer in range(L + 1)]
    W_stacked = np.zeros(((L - 1) * d + 1, d), dtype=float)
    Ws = [W_stacked[d * layer : d * (layer + 1)] for layer in range(L)]
    bs = np.zeros((L, d), dtype=float)

    with torch.no_grad():
        u = torch.zeros(model.input_dim, dtype=torch.float32)
        for layer in range(L):
            if layer > 0:
                w_hat = relu(module.W_hat_layers[layer](u))
                W = module.W_bar_layers[layer].weight @ torch.diag(w_hat)
                Ws[layer - 1][:] = W.detach().cpu().numpy()
            v_hat = module.V_hat_layers[layer](u)
            V = module.V_bar_layers[layer].weight @ torch.diag(v_hat)
            b = module.b_layers[layer](u)
            Vs[layer][:] = V.detach().cpu().numpy()
            bs[layer] = b.detach().cpu().numpy()
            u = relu(module.u_layers[layer](u))

        layer = L
        w_hat = relu(module.W_hat_layers[layer](u))
        W = module.W_bar_layers[layer].weight @ torch.diag(w_hat)
        Ws[layer - 1][:] = W.detach().cpu().numpy()
        v_hat = module.V_hat_layers[layer](u)
        V = module.V_bar_layers[layer].weight @ torch.diag(v_hat)
        Vs[-1][:] = V.detach().cpu().numpy()
        b_fin = float(module.b_layers[layer](u)[0].detach().cpu().numpy())

    return V_stacked, W_stacked, bs, b_fin


def _picnn_min_score_at_lifted(
    V_stacked: np.ndarray,
    W_stacked: np.ndarray,
    bs: np.ndarray,
    b_fin: float,
    *,
    L: int,
    hidden_dim: int,
    y_dim: int,
    epsilon: float,
) -> float:
    """Minimum PICNN score over y at fixed lifted parameters (e2e-conformal)."""
    cp = _import_cvxpy()

    sigma = cp.Variable((L, hidden_dim), nonneg=True, name="sigma")
    y = cp.Variable(y_dim, name="y")
    Vs = [V_stacked[hidden_dim * layer : hidden_dim * (layer + 1)] for layer in range(L + 1)]
    Ws = [W_stacked[hidden_dim * layer : hidden_dim * (layer + 1)] for layer in range(L)]

    constraints = []
    for layer in range(L):
        if layer == 0:
            constraints.append(sigma[layer] >= Vs[0] @ y + bs[layer])
        else:
            constraints.append(
                sigma[layer] >= Ws[layer - 1] @ sigma[layer - 1] + Vs[layer] @ y + bs[layer]
            )

    final_expr = Ws[L - 1] @ sigma[L - 1] + Vs[L] @ y + b_fin
    if epsilon == 0.0:
        objective = cp.Minimize(final_expr)
    else:
        objective = cp.Minimize(final_expr + epsilon * cp.norm(y, p="inf"))

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception:
        problem.solve(solver=cp.SCS, eps=1e-5, max_iters=20_000)
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"PICNN min-score problem status: {problem.status}")
    y_val = np.asarray(y.value, dtype=float)
    return float(problem.value), y_val


def _picnn_q_feasible_conditional(
    model: _YehPICNNModelConditional,
    x: np.ndarray,
    q: float,
) -> tuple[float, np.ndarray]:
    """Ensure q is large enough for a non-empty sublevel set at x (e2e-conformal)."""
    V, W, bs, b_fin = _picnn_lifted_parameters_conditional(model, x)
    picnn_min, y_min = _picnn_min_score_at_lifted(
        V,
        W,
        bs,
        b_fin,
        L=model.L,
        hidden_dim=model.hidden_dim,
        y_dim=model.y_dim,
        epsilon=model.epsilon,
    )
    return max(float(q), picnn_min + 1e-5), y_min


class _YehPICNNDualRobustSolver:
    """Yeh et al. PICNN robust optimization over the calibrated sublevel set.

    Uses the lifted dual formulation from problems/picnn.py in e2e-conformal,
    not a polygon outer approximation, for worst-case cost evaluation.
    """

    def __init__(
        self,
        threshold: float,
        V_stacked: np.ndarray,
        W_stacked: np.ndarray,
        bs: np.ndarray,
        b_fin: float,
        *,
        L: int,
        hidden_dim: int,
        y_dim: int,
        epsilon: float,
    ) -> None:
        self.threshold = float(threshold)
        self.V_stacked = V_stacked
        self.W_stacked = W_stacked
        self.bs = bs
        self.b_fin = float(b_fin)
        self.L = int(L)
        self.hidden_dim = int(hidden_dim)
        self.y_dim = int(y_dim)
        self.epsilon = float(epsilon)
        self._dual_cache: dict[float, Any] = {}
        self._support_problem: Any | None = None
        self._support_cache: dict[tuple[float, ...], np.ndarray] = {}

    @classmethod
    def from_conditional_model(
        cls,
        model: _YehPICNNModelConditional,
        threshold: float,
        x: np.ndarray,
    ) -> "_YehPICNNDualRobustSolver":
        q_feas, _ = _picnn_q_feasible_conditional(model, x, threshold)
        threshold = q_feas
        V, W, bs, b_fin = _picnn_lifted_parameters_conditional(model, x)
        return cls(
            threshold,
            V,
            W,
            bs,
            b_fin,
            L=model.L,
            hidden_dim=model.hidden_dim,
            y_dim=model.y_dim,
            epsilon=model.epsilon,
        )

    def _dual_problem(self, quadratic_weight: float):
        key = float(np.round(quadratic_weight, 12))
        if key in self._dual_cache:
            return self._dual_cache[key]

        cp = _import_cvxpy()

        L = self.L
        d = self.hidden_dim
        y_dim = self.y_dim
        base = cp.Parameter(y_dim, name="base")
        t = cp.Variable(name="t")
        z = cp.hstack([t, 1.0 - t])
        nu = cp.Variable(2 * L * d + 1, nonneg=True, name="nu")
        nus = [nu[d * layer : d * (layer + 1)] for layer in range(2 * L + 1)]
        mu = nu[-1]
        Ws = [self.W_stacked[d * layer : d * (layer + 1)] for layer in range(L)]

        constraints = [t >= 0.0, t <= 1.0]
        constraints.extend(
            Ws[layer].T @ nus[L + layer + 1] - nus[L + layer] - nus[layer] == 0.0
            for layer in range(L)
        )
        if self.epsilon == 0.0:
            constraints.append(self.V_stacked.T @ nu[L * d :] == z)
        else:
            rho = cp.Variable(2 * y_dim, nonneg=True, name="rho")
            constraints.extend(
                [
                    self.V_stacked.T @ nu[L * d :] + rho[:y_dim] - rho[y_dim:] == z,
                    mu == cp.sum(rho),
                ]
            )

        objective = (self.threshold - self.b_fin) * mu + base @ z
        for layer in range(L):
            objective -= self.bs[layer] @ nus[L + layer]
        if quadratic_weight > 0.0:
            objective += quadratic_weight * cp.square(2.0 * t - 1.0)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        self._dual_cache[key] = (problem, base, t)
        return self._dual_cache[key]

    def _solve_cvxpy(self, problem) -> None:
        cp = _import_cvxpy()

        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            problem.solve(solver=cp.SCS, warm_start=True, eps=1e-5, max_iters=20_000)
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"PICNN robust problem status: {problem.status}")

    def solve(self, center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
        base_vec = np.asarray(center, dtype=float) + np.asarray(cost, dtype=float)
        if len(base_vec) != 2:
            raise ValueError("Yeh PICNN robust solver currently supports 2D outcomes.")

        problem, base_param, t_var = self._dual_problem(loss_spec.quadratic_weight)
        base_param.value = base_vec
        self._solve_cvxpy(problem)
        t = float(np.clip(t_var.value, 0.0, 1.0))
        z = np.array([t, 1.0 - t])
        objective = float(problem.value)
        worst_residual = self._worst_residual_for_direction(z)
        worst_y = np.asarray(center, dtype=float) + worst_residual
        return RobustResult(
            z=z,
            worst_y=worst_y,
            objective=objective,
            worst_residual=worst_residual,
        )

    def _worst_residual_for_direction(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        try:
            return self.support_argmax(direction)
        except RuntimeError:
            return np.zeros(self.y_dim, dtype=float)

    def _build_support_problem(self):
        if self._support_problem is not None:
            return self._support_problem

        cp = _import_cvxpy()

        L = self.L
        d = self.hidden_dim
        y_dim = self.y_dim
        c_param = cp.Parameter(y_dim, name="c")
        sigma = cp.Variable((L, d), nonneg=True, name="sigma")
        y = cp.Variable(y_dim, name="y")
        Vs = [self.V_stacked[d * layer : d * (layer + 1)] for layer in range(L + 1)]
        Ws = [self.W_stacked[d * layer : d * (layer + 1)] for layer in range(L)]

        constraints = []
        for layer in range(L):
            if layer == 0:
                constraints.append(sigma[layer] >= Vs[0] @ y + self.bs[layer])
            else:
                constraints.append(
                    sigma[layer] >= Ws[layer - 1] @ sigma[layer - 1] + Vs[layer] @ y + self.bs[layer]
                )

        final_expr = Ws[L - 1] @ sigma[L - 1] + Vs[L] @ y + self.b_fin
        if self.epsilon == 0.0:
            constraints.append(final_expr <= self.threshold)
        else:
            kappa = cp.Variable(name="kappa")
            constraints.extend(
                [
                    final_expr + self.epsilon * kappa <= self.threshold,
                    kappa >= y,
                    kappa >= -y,
                ]
            )

        problem = cp.Problem(cp.Maximize(c_param @ y), constraints)
        self._support_problem = (problem, c_param, y)
        return self._support_problem

    def support_argmax(self, direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        key = tuple(np.round(direction, 8))
        if key in self._support_cache:
            return self._support_cache[key].copy()

        problem, c_param, y_var = self._build_support_problem()
        c_param.value = direction
        cp = _import_cvxpy()
        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True)
        except Exception:
            problem.solve(solver=cp.SCS, warm_start=True, eps=1e-5, max_iters=20_000)
        if problem.status in {"unbounded", "unbounded_inaccurate"}:
            raise RuntimeError(f"PICNN support problem status: {problem.status}")
        if problem.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError(f"PICNN support problem status: {problem.status}")
        value = np.asarray(y_var.value, dtype=float)
        self._support_cache[key] = value
        return value.copy()

    def support(self, direction: np.ndarray) -> float:
        argmax = self.support_argmax(direction)
        return float(np.asarray(direction, dtype=float) @ argmax)


def _convex_polygon_from_picnn(
    model: _YehPICNNModel,
    threshold: float,
    residuals: np.ndarray,
    n_dirs: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate {r: PICNN(r) <= threshold} by radial boundary sampling."""
    import torch

    residuals = np.asarray(residuals, dtype=float)
    start = residuals.mean(axis=0).astype(np.float32)
    y = torch.tensor(start[None, :], dtype=torch.float32, requires_grad=True)
    x = torch.zeros((1, 1), dtype=torch.float32)
    optimizer = torch.optim.Adam([y], lr=0.05)
    for _ in range(250):
        optimizer.zero_grad()
        score = model.module(x, y).mean()
        score.backward()
        optimizer.step()
    center = y.detach().cpu().numpy()[0].astype(float)
    center_score = float(model.scores(center[None, :])[0])
    if center_score > threshold:
        center = residuals[np.argmin(model.scores(residuals))]

    angles = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    base_radius = float(np.max(np.linalg.norm(residuals - center[None, :], axis=1)) + 1.0)
    max_radius = max(100.0, 64.0 * base_radius)
    points = []
    for direction in dirs:
        lo, hi = 0.0, max(base_radius, 1.0)
        found_boundary = False
        for _ in range(30):
            candidate = center + hi * direction
            if model.scores(candidate[None, :])[0] > threshold:
                found_boundary = True
                break
            hi *= 2.0
            if hi > max_radius:
                break
        if not found_boundary:
            raise ValueError(
                "PICNN sublevel set appears unbounded in at least one direction; "
                "increase the PICNNAdapter epsilon parameter."
            )
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            candidate = center + mid * direction
            if model.scores(candidate[None, :])[0] <= threshold:
                lo = mid
            else:
                hi = mid
        points.append(center + lo * direction)
    vertices = np.vstack(points)

    W = []
    h = []
    for i in range(len(vertices)):
        p = vertices[i]
        q = vertices[(i + 1) % len(vertices)]
        edge = q - p
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal = normal / norm
        W.append(normal)
        h.append(float(normal @ p))
    if not W:
        radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
        radius = max(radius, 1e-4)
        W, h = ball_as_halfspaces(radius)
    else:
        W = np.vstack(W)
        h = np.asarray(h)
    return W, h, vertices, center


class PICNNAdapter(BaseMethod):
    """Yeh et al. ETO baseline (FALCON PICNN + split conformal + lifted dual RO).

    Matches chrisyeh96/e2e-conformal ``run_portfolio_picnn.py`` defaults:
    hidden_dim=128, L=2, ETO epochs=50, lr=1e-2, l2reg=1e-3, MALA num_chains=50,
    y_in_output_layer=False, epsilon=0.

    Residual experiments condition the score on f̂(x) (same role as x in the
    portfolio code).  Worst-case cost uses the lifted PICNN dual formulation,
    not a polygon approximation.
    """

    name = "Yeh et al. (PICNN/FALCON)"
    color = "#bcbd22"

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 2,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-2,
        l2reg: float = 1e-3,
        num_chains: int = 50,
        num_samples: int = 1,
        burnin: int = 100,
        std: float = 0.25,
        warm_start: bool = False,
        weight_towards_zero: float = 1.0,
        boundary_dirs: int = 96,
        epsilon: float = 0.0,
        y_in_output_layer: bool = False,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2reg = l2reg
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.burnin = burnin
        self.std = std
        self.warm_start = warm_start
        self.weight_towards_zero = weight_towards_zero
        self.boundary_dirs = boundary_dirs
        self.epsilon = epsilon
        self.y_in_output_layer = y_in_output_layer

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        if context.centers_learn is None:
            raise ValueError("PICNNAdapter requires context.centers_learn (f̂ on learn split).")
        x_learn = np.asarray(context.centers_learn, dtype=np.float32)
        n = min(len(x_learn), len(r_learn))
        self.model_ = _fit_yeh_picnn_falcon_conditional(
            x_learn[:n],
            r_learn[:n],
            context,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            l2reg=self.l2reg,
            num_chains=self.num_chains,
            num_samples=self.num_samples,
            burnin=self.burnin,
            std=self.std,
            warm_start=self.warm_start,
            weight_towards_zero=self.weight_towards_zero,
            epsilon=self.epsilon,
            y_in_output_layer=self.y_in_output_layer,
        )

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        if context.centers_recal is None:
            raise ValueError("PICNNAdapter requires context.centers_recal (f̂ on recal split).")
        x_recal = np.asarray(context.centers_recal, dtype=np.float32)
        n = min(len(x_recal), len(r_recal))
        x_recal = x_recal[:n]
        r_recal = r_recal[:n]

        scores = self.model_.scores_conditional(x_recal, r_recal.astype(np.float32))
        if len(scores) == 0:
            raise ValueError("PICNNAdapter requires nonempty calibration data.")
        q = float(conformal_quantile(scores, context.alpha))
        x_mean = x_recal.mean(axis=0, keepdims=True).astype(np.float32)
        q_plot, _ = _picnn_q_feasible_conditional(
            self.model_, x_mean.reshape(-1), q
        )
        W, h, _, _ = _convex_polygon_from_picnn_conditional(
            self.model_,
            q_plot,
            x_mean,
            n_dirs=self.boundary_dirs,
            cal_residuals=r_recal,
        )
        model = self.model_
        q_val = q

        def score_fn(r: np.ndarray) -> np.ndarray:
            r_arr = np.asarray(r, dtype=float)
            single = r_arr.ndim == 1
            if single:
                r_arr = r_arr[None, :]
            out = model.scores_conditional(
                np.tile(x_mean, (len(r_arr), 1)),
                r_arr.astype(np.float32),
            )
            return float(out[0]) if single else out

        def coverage_fn(centers: np.ndarray, residuals: np.ndarray) -> np.ndarray:
            centers = np.asarray(centers, dtype=np.float32)
            residuals = np.asarray(residuals, dtype=np.float32)
            cal_scores = model.scores_conditional(centers, residuals)
            return cal_scores <= q_val + 1e-9

        def robust_solver_fn(center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
            solver = _YehPICNNDualRobustSolver.from_conditional_model(
                model, q_val, np.asarray(center, dtype=float)
            )
            if loss_spec.name == "newsvendor":
                # The lifted dual in solver.solve() is for the linear/quadratic loss.
                return _newsvendor_robust_via_support(
                    solver.support, solver._worst_residual_for_direction, center, cost, loss_spec
                )
            return solver.solve(center, cost, loss_spec)

        return SetResult(
            self.name,
            kind="polytope",
            color=self.color,
            W=W,
            h=h,
            score_fn=score_fn,
            threshold=q,
            coverage_fn=coverage_fn,
            robust_solver_fn=robust_solver_fn,
            metadata={
                "q": q,
                "q_plot": q_plot,
                "conditional_on_center": True,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "epochs": self.epochs,
                "lr": self.lr,
                "l2reg": self.l2reg,
                "num_chains": self.num_chains,
                "epsilon": self.epsilon,
                "y_in_output_layer": self.y_in_output_layer,
                "boundary_dirs": self.boundary_dirs,
                "selection": "Yeh ETO: FALCON PICNN on (f̂,r), lifted dual RO at each f̂(x)",
            },
        )


# ---------------------------------------------------------------------------
# Yeh et al. (TMLR 2025), "End-to-End Conformal Calibration for Optimization
# Under Uncertainty": end-to-end training of the PICNN score.  Port of
# chrisyeh96/e2e-conformal (models/picnn_e2e.py, problems/picnn.py,
# utils/conformal.py, run_portfolio_picnn.py) to the residual-set pipeline,
# with the PICNN conditioned on the point prediction f̂(x).
# ---------------------------------------------------------------------------


def _picnn_lifted_torch(module, x):
    """Batched lifted PICNN parameters (V, W, b, b_L) at inputs x (B, x_dim); differentiable.

    Same layer indexing as ``PICNNProblem.solve_cvxpylayers`` in e2e-conformal:
    V_stacked (B, L d + 1, n), W_stacked (B, (L-1) d + 1, d), bs (B, L, d), b_fin (B,).
    """
    import torch

    relu = torch.nn.functional.relu
    L = module.L
    u = x
    Vs, Ws, bs = [], [], []
    for layer in range(L):
        if layer > 0:
            w_hat = relu(module.W_hat_layers[layer](u))
            Ws.append(module.W_bar_layers[layer].weight[None] * w_hat[:, None, :])
        v_hat = module.V_hat_layers[layer](u)
        Vs.append(module.V_bar_layers[layer].weight[None] * v_hat[:, None, :])
        bs.append(module.b_layers[layer](u))
        u = relu(module.u_layers[layer](u))
    w_hat = relu(module.W_hat_layers[L](u))
    Ws.append(module.W_bar_layers[L].weight[None] * w_hat[:, None, :])
    v_hat = module.V_hat_layers[L](u)
    Vs.append(module.V_bar_layers[L].weight[None] * v_hat[:, None, :])
    b_fin = module.b_layers[L](u)[:, 0]
    return torch.cat(Vs, dim=1), torch.cat(Ws, dim=1), torch.stack(bs, dim=1), b_fin


def _yeh_calc_q(scores, alpha: float):
    """utils/conformal.py::calc_q -- the ceil((n+1)(1-alpha))-th smallest score (inf if too few)."""
    import torch

    n = len(scores)
    j = int(np.ceil((n + 1) * (1.0 - alpha)))
    if j > n:
        return torch.tensor(float("inf"), dtype=scores.dtype)
    return scores[torch.argsort(scores)[j - 1]]


def _yeh_picnn_problems(L: int, d: int, n: int, r_std: np.ndarray, loss_spec: LossSpec,
                        epigraph: bool = False, epsilon: float = 0.0) -> dict:
    """DPP cvxpy problems over the standardized PICNN sublevel set {y : s(x, y) <= q}.

    ``robust``: min_z max_{r in C(x)} g(z, base + r) via the dual of the inner LP
    (problems/picnn.py), where r = r_mean + r_std * y and base = c + f̂(x) + r_mean
    (linear / quadratic-regularized) or f̂(x) + r_mean (newsvendor, a maximum of
    2^n affine pieces in the outcome, each handled by its own dual block).
    ``min``: min_y s(x, y) (for the empty-set guard q >= q_min).  ``support``:
    max_y c^T y over the set.  Each problem has its own parameter objects.
    """
    import itertools

    cp = _import_cvxpy()
    r_std = np.asarray(r_std, dtype=float)

    def picnn_params():
        return {
            "V": cp.Parameter((L * d + 1, n)),
            "W": cp.Parameter(((L - 1) * d + 1, d)),
            "bs": cp.Parameter((L, d)),
            "b_fin": cp.Parameter(),
        }

    def primal_constraints(P, y):
        sigma = cp.Variable((L, d), nonneg=True)
        Vs = [P["V"][d * layer: d * (layer + 1)] for layer in range(L + 1)]
        Ws = [P["W"][d * layer: d * (layer + 1)] for layer in range(L)]
        cons = [sigma[0] >= Vs[0] @ y + P["bs"][0]]
        for layer in range(1, L):
            cons.append(sigma[layer] >= Ws[layer - 1] @ sigma[layer - 1] + Vs[layer] @ y + P["bs"][layer])
        final = cp.sum(Ws[L - 1] @ sigma[L - 1] + Vs[L] @ y + P["b_fin"])
        if epsilon > 0.0:  # s = PICNN + eps * ||y||_inf, via kappa >= |y_i|
            kappa = cp.Variable()
            cons += [kappa >= y, kappa >= -y]
            final = final + float(epsilon) * kappa
        return cons, final

    # robust decision problem
    P = picnn_params()
    P["q"] = cp.Parameter()
    P["base"] = cp.Parameter(n)

    def dual_block(c_expr):
        nu = cp.Variable(2 * L * d + 1, nonneg=True)
        nus = [nu[d * k: d * (k + 1)] for k in range(2 * L + 1)]
        Ws = [P["W"][d * layer: d * (layer + 1)] for layer in range(L)]
        cons = [Ws[layer].T @ nus[L + layer + 1] - nus[L + layer] - nus[layer] == 0 for layer in range(L)]
        if epsilon > 0.0:  # dual of the eps * ||y||_inf term: rho+ - rho- enters stationarity, sum(rho) = eps mu
            rho = cp.Variable(2 * n, nonneg=True)
            cons.append(P["V"].T @ nu[L * d:] + rho[:n] - rho[n:] == c_expr)
            cons.append(cp.sum(rho) == float(epsilon) * nu[-1])
        else:
            cons.append(P["V"].T @ nu[L * d:] == c_expr)
        obj = (P["q"] - P["b_fin"]) * nu[-1]
        for layer in range(L):
            obj = obj - P["bs"][layer] @ nus[L + layer]
        return obj, cons

    z = cp.Variable(n, nonneg=True)
    cons = [cp.sum(z) == 1]
    if loss_spec.name == "newsvendor":
        c_o, c_u = loss_spec._nv_costs(n)
        tau = cp.Variable()
        for pattern in itertools.product([0, 1], repeat=n):  # 1 = underage piece in that coordinate
            g = np.where(np.asarray(pattern) == 1, c_u, -c_o)  # piece = g^T y - g^T z
            obj_s, cons_s = dual_block(r_std * g)
            cons += cons_s
            cons.append(tau >= P["base"] @ g - z @ g + obj_s)
        objective = tau
    else:
        obj_s, cons_s = dual_block(cp.multiply(r_std, z))
        cons += cons_s
        objective = obj_s + P["base"] @ z
        if loss_spec.quadratic_weight > 0.0:
            # beta (z1 - z2)^2 via its SOC epigraph: diffcp cannot differentiate a quadratic objective.
            sq = cp.Variable(nonneg=True)
            cons.append(cp.SOC(1 + sq, cp.hstack([2 * (z[0] - z[1]), 1 - sq])))
            objective = objective + float(loss_spec.quadratic_weight) * sq
    if epigraph:  # min v s.t. v >= objective: v* is the robust value, as a layer output
        v = cp.Variable()
        cons.append(v >= objective)
        robust = cp.Problem(cp.Minimize(v), cons)
        out = {"robust": (robust, P, z), "value": v}
    else:
        robust = cp.Problem(cp.Minimize(objective), cons)
        out = {"robust": (robust, P, z)}
    assert robust.is_dpp()

    Pm = picnn_params()
    y = cp.Variable(n)
    cons, final = primal_constraints(Pm, y)
    out["min"] = (cp.Problem(cp.Minimize(final), cons), Pm, y)
    assert out["min"][0].is_dpp()

    Ps = picnn_params()
    Ps["q"] = cp.Parameter()
    Ps["c"] = cp.Parameter(n)
    y = cp.Variable(n)
    cons, final = primal_constraints(Ps, y)
    cons.append(final <= Ps["q"])
    out["support"] = (cp.Problem(cp.Maximize(Ps["c"] @ y), cons), Ps, y)
    assert out["support"][0].is_dpp()
    return out


# SCS with its default tolerances inside cvxpylayers, as in e2e-conformal.  (An exact
# interior-point solve puts z* at an LP vertex, where diffcp returns ~zero gradients.)
# YEH_LAYER_JOBS solves the batch in parallel; results do not depend on it.
_YEH_LAYER_JOBS = int(__import__("os").environ.get("YEH_LAYER_JOBS", "1"))
_YEH_LAYER_ARGS = {"n_jobs_forward": _YEH_LAYER_JOBS, "n_jobs_backward": _YEH_LAYER_JOBS}


def _yeh_realized_loss_torch(z, y, cost, loss_spec: LossSpec):
    """Task loss at the realized outcome (Yeh et al. train on f(x, y, z*(x)))."""
    import torch

    if loss_spec.name == "newsvendor":
        c_o, c_u = (torch.tensor(a, dtype=z.dtype) for a in loss_spec._nv_costs(z.shape[1]))
        return (c_o * torch.clamp(z - y, min=0.0) + c_u * torch.clamp(y - z, min=0.0)).sum(-1)
    out = ((cost[None] + y) * z).sum(-1)
    if loss_spec.quadratic_weight > 0.0:
        out = out + float(loss_spec.quadratic_weight) * (z[:, 0] - z[:, 1]) ** 2
    return out


def _solve_cvxpy_lp(problem) -> None:
    cp = _import_cvxpy()
    try:
        problem.solve(solver=cp.ECOS, max_iters=500, abstol=1e-9, reltol=1e-9, feastol=1e-9)
    except Exception:
        problem.solve(solver=cp.SCS, eps=1e-7, max_iters=100_000)
    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"Yeh PICNN problem status: {problem.status}")


class YehE2EAdapter(BaseMethod):
    """Yeh et al. end-to-end conformal calibration with a PICNN score s(f̂(x), r).

    1. Standardize the conditioning input f̂(x) and the residual with statistics of the
       training part of D_n; hold out ``val_frac`` of D_n for early stopping (the role
       of the calibration loader in e2e-conformal, so that D_m stays untouched).
    2. ETO: fit the PICNN energy by FALCON/MALA (50 epochs, Adam lr 1e-2, weight decay 1e-3).
    3. E2E (initialised from ETO): per minibatch, half the points set q by the exact
       conformal quantile of their scores, q is raised to the per-x minimum score
       (no gradient), z*(x) comes from the dual LP through a cvxpylayer for the other
       half, and the loss is their realized task loss + 0.01 q^2.  Adam (lr ``e2e_lr``,
       weight decay 1e-3), weights clamped after each step, at most 100 epochs, early
       stopping with patience 10 on the validation task loss.
    4. Final split-conformal threshold on D_m; robust decisions from the exact dual LP.
    """

    name = "Yeh et al. (PICNN E2E)"
    color = "#bcbd22"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        eto_epochs: int = 50,
        eto_lr: float = 1e-2,
        l2reg: float = 1e-3,
        e2e_lrs: tuple[float, ...] = (1e-3, 1e-4),
        max_e2e_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        large_q_penalty: float = 1e-2,
        val_frac: float = 0.2,
        area_dirs: int = 128,
        task_loss: str = "realized",
        name: str | None = None,
        epsilon: float = 1.0,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.eto_epochs = eto_epochs
        self.eto_lr = eto_lr
        self.l2reg = l2reg
        self.e2e_lrs = tuple(e2e_lrs)
        self.max_e2e_epochs = max_e2e_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.large_q_penalty = large_q_penalty
        self.val_frac = val_frac
        self.area_dirs = area_dirs
        # "realized": f(x, y, z*(x)) at the true outcome (Yeh et al., Alg. 1).  "robust": the
        # robust objective max_{y in C(x)} f(x, y, z*(x)) instead (our evaluation metric).
        if task_loss not in {"realized", "robust"}:
            raise ValueError(task_loss)
        self.task_loss = task_loss
        # epsilon > 0: compact sublevel sets s + eps * ||y||_inf (Yeh et al., App. C.4).  With
        # epsilon = 0 (their experiments) the learned sets were unbounded on Energy Efficiency.
        self.epsilon = float(epsilon)
        if name is not None:
            self.name = name

    # standardization ----------------------------------------------------------
    def _sx(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=float) - self.x_mean_) / self.x_std_).astype(np.float32)

    def _sr(self, r: np.ndarray) -> np.ndarray:
        return ((np.asarray(r, dtype=float) - self.r_mean_) / self.r_std_).astype(np.float32)

    def _scores(self, centers: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            self.model_.module.eval()
            out = self.model_.module(torch.from_numpy(self._sx(centers)), torch.from_numpy(self._sr(residuals)))
        return out.numpy().astype(float)

    def _lifted_numpy(self, center: np.ndarray):
        import torch

        with torch.no_grad():
            V, W, bs, b_fin = _picnn_lifted_torch(self.model_.module, torch.from_numpy(self._sx(np.atleast_2d(center))))
        return V[0].double().numpy(), W[0].double().numpy(), bs[0].double().numpy(), float(b_fin[0])

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        import io
        import time as _time

        import torch

        _import_cvxpy()  # SciPy shim must run before cvxpylayers imports cvxpy
        from cvxpylayers.torch import CvxpyLayer

        _ensure_2d(context)
        if context.centers_learn is None:
            raise ValueError("YehE2EAdapter requires context.centers_learn (f̂ on learn split).")
        loss_spec = _loss(context)
        centers = np.asarray(context.centers_learn, dtype=float)
        residuals = np.asarray(r_learn, dtype=float)
        n = min(len(centers), len(residuals))
        centers, residuals = centers[:n], residuals[:n]
        rng = np.random.default_rng(context.seed + 1931)
        perm = rng.permutation(n)
        n_val = int(round(self.val_frac * n))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        self.x_mean_, self.x_std_ = centers[tr_idx].mean(axis=0), centers[tr_idx].std(axis=0) + 1e-8
        self.r_mean_, self.r_std_ = residuals[tr_idx].mean(axis=0), residuals[tr_idx].std(axis=0) + 1e-8
        xs, rs = self._sx(centers), self._sr(residuals)

        t0 = _time.perf_counter()
        self.model_ = _fit_yeh_picnn_falcon_conditional(
            xs[tr_idx], rs[tr_idx], context,
            hidden_dim=self.hidden_dim, n_layers=self.n_layers,
            epochs=self.eto_epochs, lr=self.eto_lr, l2reg=self.l2reg, epsilon=self.epsilon,
        )
        self.eto_time_ = _time.perf_counter() - t0

        module = self.model_.module
        L, d, ny = self.n_layers, self.hidden_dim, residuals.shape[1]
        use_value = self.task_loss == "robust"
        probs = _yeh_picnn_problems(L, d, ny, self.r_std_, loss_spec, epigraph=use_value, epsilon=self.epsilon)
        rob, P, z_var = probs["robust"]
        robust_layer = CvxpyLayer(rob, parameters=[P["V"], P["W"], P["bs"], P["b_fin"], P["q"], P["base"]],
                                  variables=[z_var, probs["value"]] if use_value else [z_var])
        mn, Pm, y_var = probs["min"]
        min_layer = CvxpyLayer(mn, parameters=[Pm["V"], Pm["W"], Pm["bs"], Pm["b_fin"]], variables=[y_var])

        cost = torch.tensor(_cost(context))
        offset = _cost(context) if loss_spec.name != "newsvendor" else 0.0
        base_all = torch.tensor(offset + centers + self.r_mean_)
        y_all = torch.tensor(centers + residuals)
        x_t, r_t = torch.from_numpy(xs), torch.from_numpy(rs)

        def decisions(idx, q):
            V, W, bs, b_fin = _picnn_lifted_torch(module, x_t[idx])
            V, W, bs, b_fin = V.double(), W.double(), bs.double(), b_fin.double()
            with torch.no_grad():  # empty-set guard; no gradient through q_min (problems/picnn.py)
                y_min, = min_layer(V.detach(), W.detach(), bs.detach(), b_fin.detach(), solver_args=_YEH_LAYER_ARGS)
                q_min = module(x_t[idx], y_min.float()).double() + 1e-5
            q_feas = torch.maximum(q.double().expand(len(idx)), q_min)
            out = robust_layer(V, W, bs, b_fin, q_feas, base_all[idx], solver_args=_YEH_LAYER_ARGS)
            return out[0], (out[1] if use_value else None)

        def batch_task_loss(idx, q):
            z, value = decisions(idx, q)
            if use_value:
                return value.reshape(-1)
            return _yeh_realized_loss_torch(z, y_all[idx], cost, loss_spec)

        def val_loss() -> float:
            with torch.no_grad():
                module.eval()
                q = _yeh_calc_q(module(x_t[val_idx], r_t[val_idx]).double(), context.alpha)
                return float(batch_task_loss(val_idx, q).mean())

        eto_buf = io.BytesIO()
        torch.save(module.state_dict(), eto_buf)
        t0 = _time.perf_counter()
        runs = []
        for lr in self.e2e_lrs:
            eto_buf.seek(0)
            module.load_state_dict(torch.load(eto_buf))
            run_rng = np.random.default_rng(context.seed + 1931 + int(round(-np.log10(lr))))
            runs.append((lr, *self._train_e2e(module, lr, run_rng, tr_idx, batch_task_loss, val_loss, x_t, r_t,
                                               context.alpha)))
        lr, best, best_epoch, history, state = min(runs, key=lambda item: item[1])
        module.load_state_dict(state)
        self.model_.eval()
        self.e2e_time_ = _time.perf_counter() - t0
        self.history_ = {run[0]: run[3] for run in runs}
        self.e2e_info_ = {"e2e_lr": lr, "best_epoch": best_epoch, "best_val_task_loss": best,
                          "epochs_run": len(history),
                          "val_by_lr": {run[0]: run[1] for run in runs}}
        self.problems_ = _yeh_picnn_problems(L, d, ny, self.r_std_, loss_spec, epsilon=self.epsilon)

    def _train_e2e(self, module, lr, rng, tr_idx, batch_task_loss, val_loss, x_t, r_t, alpha):
        """train_e2e of e2e-conformal: returns (best val loss, best epoch, history, best state dict)."""
        import io

        import torch

        optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=self.l2reg)
        buf = io.BytesIO()
        torch.save(module.state_dict(), buf)
        best, best_epoch, bad, history = float("inf"), -1, 0, []
        import cvxpy.error
        import diffcp.cone_program

        try:
            for epoch in range(self.max_e2e_epochs):
                order = tr_idx[rng.permutation(len(tr_idx))]
                total, count = 0.0, 0
                for start in range(0, len(order), self.batch_size):
                    batch = order[start: start + self.batch_size]
                    if len(batch) < 4:
                        continue
                    split = rng.permutation(len(batch))
                    cal, task = batch[split[: len(batch) // 2]], batch[split[len(batch) // 2:]]
                    module.eval()
                    q = _yeh_calc_q(module(x_t[cal], r_t[cal]).double(), alpha)
                    if not torch.isfinite(q):
                        continue
                    task_loss = batch_task_loss(task, q).mean()
                    loss = task_loss + self.large_q_penalty * q ** 2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    self.model_.clamp_weights()
                    total += float(task_loss) * len(task)
                    count += len(task)
                val = val_loss()
                history.append((epoch, total / max(count, 1), val))
                if val < best:
                    best, best_epoch, bad = val, epoch, 0
                    buf = io.BytesIO()
                    torch.save(module.state_dict(), buf)
                else:
                    bad += 1
                    if bad > self.patience:
                        break
        except (diffcp.cone_program.SolverError, cvxpy.error.SolverError):
            # e2e-conformal stops training on solver errors and keeps the best checkpoint.
            self.solver_errors_ = getattr(self, "solver_errors_", 0) + 1
        buf.seek(0)
        return best, best_epoch, history, torch.load(buf)

    # evaluation helpers -------------------------------------------------------
    def _set_params(self, P: dict, center: np.ndarray) -> None:
        V, W, bs, b_fin = self._lifted_numpy(center)
        P["V"].value, P["W"].value, P["bs"].value, P["b_fin"].value = V, W, bs, b_fin

    def _q_min_and_argmin(self, center: np.ndarray) -> tuple[float, np.ndarray]:
        problem, P, y = self.problems_["min"]
        self._set_params(P, center)
        _solve_cvxpy_lp(problem)
        return float(problem.value), np.asarray(y.value, dtype=float)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        import torch

        if context.centers_recal is None:
            raise ValueError("YehE2EAdapter requires context.centers_recal (f̂ on recal split).")
        centers_recal = np.asarray(context.centers_recal, dtype=float)
        n = min(len(centers_recal), len(r_recal))
        centers_recal, r_recal = centers_recal[:n], np.asarray(r_recal, dtype=float)[:n]
        q = float(conformal_quantile(self._scores(centers_recal, r_recal), context.alpha))
        adapter = self

        def q_feas(center: np.ndarray) -> float:
            q_min, _ = adapter._q_min_and_argmin(center)
            return max(q, q_min + 1e-5)

        def support_argmax(center: np.ndarray, qf: float, a: np.ndarray) -> np.ndarray:
            problem, P, y = adapter.problems_["support"]
            adapter._set_params(P, center)
            P["q"].value = qf
            P["c"].value = adapter.r_std_ * np.asarray(a, dtype=float)
            _solve_cvxpy_lp(problem)
            return adapter.r_mean_ + adapter.r_std_ * np.asarray(y.value, dtype=float)

        def robust_solver_fn(center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
            center = np.asarray(center, dtype=float)
            qf = q_feas(center)
            if loss_spec.name == "newsvendor":
                def sup(a):
                    return float(np.asarray(a, dtype=float) @ support_argmax(center, qf, a))
                return _newsvendor_robust_via_support(
                    sup, lambda a: support_argmax(center, qf, a), center, cost, loss_spec)
            problem, P, z = adapter.problems_["robust"]
            adapter._set_params(P, center)
            P["q"].value = qf
            P["base"].value = np.asarray(cost, dtype=float) + center + adapter.r_mean_
            _solve_cvxpy_lp(problem)
            z_val = np.clip(np.asarray(z.value, dtype=float), 0.0, None)
            z_val = z_val / max(float(z_val.sum()), 1e-12)
            worst_r = support_argmax(center, qf, z_val)
            return RobustResult(z=z_val, worst_y=center + worst_r, objective=float(problem.value),
                                worst_residual=worst_r)

        def coverage_fn(centers: np.ndarray, residuals: np.ndarray) -> np.ndarray:
            return adapter._scores(centers, residuals) <= q + 1e-9

        def score_fn(r: np.ndarray):
            r_arr = np.atleast_2d(np.asarray(r, dtype=float))
            out = adapter._scores(np.tile(centers_recal.mean(axis=0), (len(r_arr), 1)), r_arr)
            return float(out[0]) if np.ndim(r) == 1 else out

        # Volume: E_x vol(C(x)), averaged over the recalibration contexts (radial bisection
        # from the minimum-score point; the sets are convex, hence star-shaped).
        def volume_fn() -> float:
            angles = np.linspace(0.0, 2.0 * np.pi, adapter.area_dirs, endpoint=False)
            dirs = np.column_stack([np.cos(angles), np.sin(angles)])
            areas = []
            module = adapter.model_.module
            for center in centers_recal:
                q_min, y_min = adapter._q_min_and_argmin(center)
                qf = max(q, q_min + 1e-5)
                x_rep = torch.from_numpy(np.tile(adapter._sx(center[None]), (len(dirs), 1)))
                lo = np.zeros(len(dirs))
                hi = np.ones(len(dirs))
                with torch.no_grad():
                    def inside(radii):
                        pts = (y_min[None] + radii[:, None] * dirs).astype(np.float32)
                        return module(x_rep, torch.from_numpy(pts)).numpy() <= qf
                    for _ in range(40):
                        grow = inside(hi)
                        if not grow.any():
                            break
                        hi[grow] *= 2.0
                    for _ in range(50):
                        mid = 0.5 * (lo + hi)
                        ok = inside(mid)
                        lo = np.where(ok, mid, lo)
                        hi = np.where(ok, hi, mid)
                rad = lo * 1.0
                areas.append(0.5 * np.sin(2.0 * np.pi / len(dirs)) * float(np.sum(rad * np.roll(rad, -1))))
            return float(np.mean(areas) * adapter.r_std_[0] * adapter.r_std_[1])

        return SetResult(
            self.name,
            kind="picnn",
            color=self.color,
            score_fn=score_fn,
            threshold=q,
            robust_solver_fn=robust_solver_fn,
            coverage_fn=coverage_fn,
            volume_fn=volume_fn,
            metadata={
                "q": q,
                "conditional_on_center": True,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "eto_epochs": self.eto_epochs,
                "eto_lr": self.eto_lr,
                "e2e_lrs": self.e2e_lrs,
                "l2reg": self.l2reg,
                "val_frac": self.val_frac,
                "task_loss": self.task_loss,
                "epsilon": self.epsilon,
                "eto_time": self.eto_time_,
                "e2e_time": self.e2e_time_,
                **self.e2e_info_,
                "selection": "Yeh E2E: ETO (FALCON) init, E2E task-loss fine-tuning, split conformal on D_m",
            },
        )


class PCPAdapter(BaseMethod):
    """PCP (Wang et al., 2023), high-density version: union of balls around conditional samples.

    Follows the authors' setting for two-dimensional targets.  A mixture density
    network (MixD: 10 diagonal-Gaussian components, two tanh hidden layers of 16
    units) models the residual r = y - f_hat(x) given f_hat(x).  It is fitted by
    maximum likelihood (Adam, at most 2000 full-batch epochs) on D_n, with early
    stopping on a held-out fifth.  For a covariate x, PCP draws K = 1000 samples
    r_1..r_K ~ p(r | f_hat(x)) and scores s(x, r) = min_k ||r - r_k||_2.  With
    ``hd_beta`` > 0 (default 0.2, the filtering ratio the authors recommend) this
    is HD-PCP: draw ceil(K / (1 - beta)) samples and keep the K with the highest
    estimated density, which removes the far-tail samples.  The
    radius q is the split-conformal quantile of the scores on D_m, so the set
    C(x) is the union of K balls of radius q around the samples.

    The samples are a deterministic function of f_hat(x), so coverage, robust
    decisions and volume all use the same set C(x).  The losses are convex in y,
    so the worst case over the union equals the worst case over its convex hull,
    whose support function is max_k a^T r_k + q ||a||.  Volume is the area of
    the union (Monte Carlo), averaged over the calibration covariates.
    """

    name = "PCP"
    color = "#8B5CF6"

    def __init__(self, n_samples: int = 1000, n_components: int = 10, hidden: tuple[int, ...] = (16, 16),
                 max_epochs: int = 2000, lr: float = 5e-3, val_frac: float = 0.2, patience: int = 200,
                 area_points: int = 20000, hd_beta: float = 0.2) -> None:
        self.n_samples = int(n_samples)
        self.n_components = int(n_components)
        self.hidden = tuple(int(h) for h in hidden)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.val_frac = float(val_frac)
        self.patience = int(patience)
        self.area_points = int(area_points)
        self.hd_beta = float(hd_beta)

    # -- mixture density network -------------------------------------------------
    def _params(self, x_std):
        out = self.net_(x_std)
        M, d = self.n_components, self.d_
        logits = out[:, :M]
        mu = out[:, M:M + M * d].reshape(-1, M, d)
        log_sigma = out[:, M + M * d:].reshape(-1, M, d).clamp(-7.0, 3.0)
        return logits, mu, log_sigma

    def _nll(self, x_std, r_std):
        import torch

        logits, mu, log_sigma = self._params(x_std)
        z = (r_std[:, None, :] - mu) / torch.exp(log_sigma)
        log_comp = -0.5 * (z ** 2).sum(-1) - log_sigma.sum(-1) - 0.5 * self.d_ * float(np.log(2.0 * np.pi))
        return -torch.logsumexp(torch.log_softmax(logits, dim=1) + log_comp, dim=1).mean()

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        import copy

        import torch
        from torch import nn

        _ensure_2d(context)
        if context.centers_learn is None:
            raise ValueError("PCPAdapter requires context.centers_learn (f_hat on the learning split).")
        x = np.asarray(context.centers_learn, dtype=float)
        r = np.asarray(r_learn, dtype=float)
        n = min(len(x), len(r))
        x, r = x[:n], r[:n]
        self.d_ = r.shape[1]
        rng = np.random.default_rng(int(context.seed) + 5501)
        perm = rng.permutation(n)
        n_val = max(1, int(round(self.val_frac * n)))
        val, tr = perm[:n_val], perm[n_val:]
        self.x_mean_, self.x_std_ = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-8
        self.r_mean_, self.r_std_ = r[tr].mean(axis=0), r[tr].std(axis=0) + 1e-8

        def tens(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))

        x_tr, r_tr = tens((x[tr] - self.x_mean_) / self.x_std_), tens((r[tr] - self.r_mean_) / self.r_std_)
        x_va, r_va = tens((x[val] - self.x_mean_) / self.x_std_), tens((r[val] - self.r_mean_) / self.r_std_)
        torch.manual_seed(int(context.seed) + 5502)
        layers, prev = [], x.shape[1]
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.Tanh()]
            prev = width
        layers.append(nn.Linear(prev, self.n_components * (1 + 2 * self.d_)))
        self.net_ = nn.Sequential(*layers).double()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        best, best_state, bad, epochs = float("inf"), None, 0, 0
        for epoch in range(self.max_epochs):
            opt.zero_grad()
            loss = self._nll(x_tr, r_tr)
            loss.backward()
            opt.step()
            epochs = epoch + 1
            with torch.no_grad():
                val_nll = float(self._nll(x_va, r_va))
            if val_nll < best - 1e-6:
                best, best_state, bad = val_nll, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.net_.eval()
        self.sample_seed_ = int(context.seed) + 5503
        self.fit_info_ = {"mdn_epochs": epochs, "mdn_best_val_nll": best, "mdn_train_size": int(len(tr))}
        self._cache: dict[bytes, np.ndarray] = {}

    def _samples(self, center: np.ndarray) -> np.ndarray:
        """K residual samples from p(r | f_hat(x) = center); deterministic in center."""
        import torch

        key = np.ascontiguousarray(center, dtype=np.float64).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        x_std = (np.frombuffer(key, dtype=np.float64) - self.x_mean_) / self.x_std_
        with torch.no_grad():
            logits, mu, log_sigma = self._params(torch.from_numpy(x_std[None]))
        logits = logits[0].numpy()
        p = np.exp(logits - logits.max())
        p /= p.sum()
        mu, sigma = mu[0].numpy(), np.exp(log_sigma[0].numpy())
        n_draw = int(np.ceil(self.n_samples / (1.0 - self.hd_beta))) if self.hd_beta > 0 else self.n_samples
        rng = np.random.default_rng([self.sample_seed_, *np.frombuffer(key, dtype=np.uint32).tolist()])
        comp = rng.choice(self.n_components, size=n_draw, p=p)
        s_std = mu[comp] + sigma[comp] * rng.standard_normal((n_draw, self.d_))
        if n_draw > self.n_samples:
            # HD-PCP: keep the K samples with the highest estimated density.
            z = (s_std[:, None, :] - mu[None]) / sigma[None]
            log_dens = np.logaddexp.reduce(
                np.log(p)[None] - 0.5 * (z ** 2).sum(-1) - np.log(sigma).sum(-1)[None], axis=1)
            s_std = s_std[np.argsort(-log_dens, kind="stable")[: self.n_samples]]
        out = self.r_mean_ + self.r_std_ * s_std
        self._cache[key] = out
        return out

    def _min_dist(self, center: np.ndarray, r: np.ndarray) -> float:
        return float(np.min(np.linalg.norm(self._samples(center) - np.asarray(r, dtype=float), axis=1)))

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        if context.centers_recal is None:
            raise ValueError("PCPAdapter requires context.centers_recal (f_hat on the recalibration split).")
        centers_recal = np.asarray(context.centers_recal, dtype=float)
        n = min(len(centers_recal), len(r_recal))
        centers_recal, r_recal = centers_recal[:n], np.asarray(r_recal, dtype=float)[:n]
        scores = np.array([self._min_dist(c, r) for c, r in zip(centers_recal, r_recal)])
        q = float(conformal_quantile(scores, context.alpha))
        adapter = self
        center_mean = centers_recal.mean(axis=0)

        def support(samples: np.ndarray, a: np.ndarray) -> float:
            a = np.asarray(a, dtype=float)
            return float(np.max(samples @ a)) + q * float(np.linalg.norm(a))

        def support_argmax(samples: np.ndarray, a: np.ndarray) -> np.ndarray:
            a = np.asarray(a, dtype=float)
            nrm = float(np.linalg.norm(a))
            push = q * a / nrm if nrm > 1e-12 else np.zeros_like(a)
            return samples[int(np.argmax(samples @ a))] + push

        def robust_solver_fn(center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec) -> RobustResult:
            center = np.asarray(center, dtype=float)
            cost = np.asarray(cost, dtype=float)
            samples = adapter._samples(center)
            if loss_spec.name == "newsvendor":
                return _newsvendor_robust_via_support(
                    lambda a: support(samples, a), lambda a: support_argmax(samples, a), center, cost, loss_spec)
            base = center + cost

            def obj(t: float) -> float:
                z = np.array([t, 1.0 - t])
                return float(base @ z) + support(samples, z) + loss_spec.penalty(z)

            # obj is convex in t (support function plus a convex penalty): golden section.
            lo, hi = 0.0, 1.0
            inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
            c, d = hi - inv_phi * (hi - lo), lo + inv_phi * (hi - lo)
            fc, fd = obj(c), obj(d)
            for _ in range(100):
                if fc <= fd:
                    hi, d, fd = d, c, fc
                    c = hi - inv_phi * (hi - lo)
                    fc = obj(c)
                else:
                    lo, c, fc = c, d, fd
                    d = lo + inv_phi * (hi - lo)
                    fd = obj(d)
            t_best = min([0.0, 1.0, 0.5 * (lo + hi)], key=obj)
            z = np.array([t_best, 1.0 - t_best])
            worst_r = support_argmax(samples, z)
            worst_y = center + worst_r
            return RobustResult(z=z, worst_y=worst_y, objective=loss_spec.loss(worst_y, cost, z),
                                worst_residual=worst_r)

        def coverage_fn(centers: np.ndarray, residuals: np.ndarray) -> np.ndarray:
            return np.array([adapter._min_dist(c, r) <= q + 1e-9 for c, r in zip(centers, residuals)])

        def score_fn(r: np.ndarray):
            r_arr = np.atleast_2d(np.asarray(r, dtype=float))
            out = np.array([adapter._min_dist(center_mean, ri) for ri in r_arr])
            return float(out[0]) if np.ndim(r) == 1 else out

        def volume_fn() -> float:
            # Area of the union of K balls (unbiased Monte Carlo): draw a ball uniformly,
            # a point uniformly in it, and weight by 1 / (number of balls covering it).
            from scipy.spatial import cKDTree

            rng = np.random.default_rng(adapter.sample_seed_ + 1)
            m = adapter.area_points
            areas = []
            for center in centers_recal:
                samples = adapter._samples(center)
                k = rng.integers(0, len(samples), m)
                rad = q * np.sqrt(rng.random(m))
                ang = rng.uniform(0.0, 2.0 * np.pi, m)
                pts = samples[k] + rad[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])
                counts = cKDTree(samples).query_ball_point(pts, q * (1.0 + 1e-9), return_length=True)
                areas.append(len(samples) * np.pi * q ** 2 * float(np.mean(1.0 / np.maximum(counts, 1))))
            return float(np.mean(areas))

        return SetResult(
            self.name,
            kind="union_of_balls",
            color=self.color,
            score_fn=score_fn,
            threshold=q,
            robust_solver_fn=robust_solver_fn,
            coverage_fn=coverage_fn,
            volume_fn=volume_fn,
            metadata={
                "q": q,
                "K_samples": self.n_samples,
                "hd_beta": self.hd_beta,
                "n_components": self.n_components,
                "hidden": self.hidden,
                "conditional_on": "f_hat(x)",
                **self.fit_info_,
                "selection": "HD-PCP: MDN for r | f_hat(x) on D_n, K top-density samples per x, union of balls, split conformal on D_m",
            },
        )


class YehConditionalAdapter(BaseMethod):
    """Yeh et al. PICNN conditioned on the predictor output ŷ = f̂(x).

    Unlike PICNNAdapter (constant x=0), this variant trains the PICNN score
    s(ŷ, r) on (ŷ_learn, r_learn) pairs, so the uncertainty set C(ŷ) = {r : s(ŷ,r) ≤ q}
    varies with the prediction ŷ(x).  At test time, the robust decision re-evaluates
    the polygon boundary at each test-point prediction, capturing covariate shift in set
    shape.  Requires context.centers_learn to be populated by the runner.
    """

    name = "Yeh et al. (conditional)"
    color = "#d62728"  # red, distinct from the fixed-x adaptation

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2, epochs: int = 50,
                 boundary_dirs: int = 64, epsilon: float = 1.0) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.boundary_dirs = boundary_dirs
        self.epsilon = epsilon

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        if context.centers_learn is None:
            raise ValueError("YehConditionalAdapter requires context.centers_learn.")
        x_learn = np.asarray(context.centers_learn, dtype=np.float32)
        n = min(len(x_learn), len(r_learn))
        self.model_ = _fit_yeh_picnn_falcon_conditional(
            x_learn[:n], r_learn[:n], context,
            hidden_dim=self.hidden_dim, n_layers=self.n_layers, epochs=self.epochs,
            epsilon=self.epsilon,
        )
        self.x_recal_mean_ = x_learn[:n].mean(axis=0)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        # Score at the mean ŷ for threshold calibration (conservative approximation).
        # Each residual is scored against a fixed "typical" prediction.
        x_mean = self.x_recal_mean_
        scores = self.model_.scores_conditional(
            np.tile(x_mean, (len(r_recal), 1)).astype(np.float32), r_recal
        )
        q = float(conformal_quantile(scores, context.alpha))
        model = self.model_
        boundary_dirs = self.boundary_dirs

        def score_fn(r: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            single = r.ndim == 1
            if single:
                r = r[None]
            x_in = np.tile(x_mean, (len(r), 1)).astype(np.float32)
            out = model.scores_conditional(x_in, r.astype(np.float32))
            return float(out[0]) if single else out

        def robust_solver_fn(center: np.ndarray, cost: np.ndarray, loss_spec) -> object:
            from experiments.robust import RobustResult
            center = np.asarray(center, dtype=float)
            cost = np.asarray(cost, dtype=float)
            # Compute the polygon at this specific ŷ=center.
            x_in = np.asarray(center, dtype=np.float32)[None]
            W, h, _, _ = _convex_polygon_from_picnn_conditional(model, q, x_in, n_dirs=boundary_dirs)
            base = center + cost
            from experiments.sets import polytope_vertices
            verts = polytope_vertices(W, h)
            # min_z max_{v in verts} (base + v)^T z  — solved by candidates
            from experiments.robust import _linear_candidates
            candidates = _linear_candidates(verts) + [0.0, 1.0]
            best_t, best_val = 0.5, float("inf")
            for t in candidates:
                z = np.array([t, 1.0 - t])
                val = float(base @ z) + float(np.max(verts @ z)) + loss_spec.penalty(z)
                if val < best_val:
                    best_val, best_t = val, t
            z = np.array([best_t, 1.0 - best_t])
            worst_idx = int(np.argmax(verts @ z))
            worst_r = verts[worst_idx]
            worst_y = center + worst_r
            return RobustResult(z=z, worst_y=worst_y, objective=loss_spec.loss(worst_y, cost, z),
                                worst_residual=worst_r)

        W_mean, h_mean, _, _ = _convex_polygon_from_picnn_conditional(
            model, q, np.tile(x_mean, (1, 1)).astype(np.float32), n_dirs=boundary_dirs
        )
        return SetResult(
            self.name, kind="polytope", color=self.color,
            W=W_mean, h=h_mean, score_fn=score_fn, threshold=q,
            robust_solver_fn=robust_solver_fn,
            metadata={"q": q, "hidden_dim": self.hidden_dim, "n_layers": self.n_layers,
                      "conditional_on": "f_hat(x)"},
        )


class FCROMSAdapter(BaseMethod):
    color = "#2ca02c"

    def __init__(self, n_random_lines: int = 10) -> None:
        self.n_random_lines = int(n_random_lines)
        self.name = f"F-CROMS ({self.n_random_lines} random lines)"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        self.selected_ = _select_min_robust(
            _random_cut_templates(context, self.n_random_lines),
            r_learn,
            context,
        )

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.selected_["W"],
            r_recal,
            context,
            {
                "selection": f"{self.n_random_lines} random K-cut sets, robust-objective selection",
                "n_random_lines": self.n_random_lines,
                "selected_angles": self.selected_["angles"],
                "learn_q": self.selected_["learn_q"],
                "learn_robust_objective": self.selected_["learn_robust_objective"],
            },
        )


class DecisionAgnosticPolytope(BaseMethod):
    name = "Decision-agnostic polytope"
    color = "#9467bd"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        cov = np.cov(r_learn.T) + 1e-6 * np.eye(context.y_dim)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        whitening = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        self.W_ = directions(context.k_poly) @ whitening

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.W_,
            r_recal,
            context,
            {"selection": "residual PCA geometry"},
        )


class DecisionAwareNoRecal(BaseMethod):
    """Ablation of ``Ours``: the same learned geometry theta_hat(D_n), but the
    radius is set on the learning split D_n itself (no independent
    recalibration).  The geometry is shared with ``Ours`` through the fit cache,
    so the two rows differ only in the calibration split."""

    name = "Decision-aware polytope (no recal)"
    color = "#ff7f0e"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        self.geometry_, self.cached_ = _fit_ours_geometry(r_learn, context)
        W, b = self.geometry_["W"], self.geometry_["b"]
        self.h_, self.q_ = calibrate_biased_template(W, b, r_learn, context.alpha)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return SetResult(
            self.name,
            kind="polytope",
            color=self.color,
            W=self.geometry_["W"],
            h=self.h_,
            metadata={
                "q": self.q_,
                "K": self.geometry_["W"].shape[0],
                "selection": f"Ours geometry ({self.geometry_['solver']}), calibrated on learning split only",
                **_ours_metadata(self.geometry_, self.cached_),
            },
        )


class SizeAwarePolytope(BaseMethod):
    name = "Size-aware conformal polytope"
    color = "#8c564b"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        self.selected_ = _select_min_area(_template_grid(context), r_learn, context)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.selected_["W"],
            r_recal,
            context,
            {
                "selection": "minimum area on learning split",
                "selected_phi": self.selected_["phi"],
                "selected_rho": self.selected_["rho"],
                "learn_q": self.selected_["learn_q"],
                "learn_area": self.selected_["learn_area"],
            },
        )


class OursDecisionAwarePolytope(BaseMethod):
    name = "Ours (CCG grid)"
    color = "#d62728"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        if context.y_dim == 2:
            templates = _template_grid(context)
        else:
            templates = _paired_random_cut_templates(context, n_templates=max(context.n_croms, 24))
        self.selected_ = _select_min_robust(templates, r_learn, context)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        metadata = {
            "selection": "decision-aware grid with final recalibration" if context.y_dim == 2
            else "decision-aware paired-random K-cut library with final recalibration",
            "learn_q": self.selected_["learn_q"],
            "learn_robust_objective": self.selected_["learn_robust_objective"],
        }
        if "phi" in self.selected_:
            metadata.update({"selected_phi": self.selected_["phi"], "selected_rho": self.selected_["rho"]})
        return _set_from_template(
            self.name,
            self.color,
            self.selected_["W"],
            r_recal,
            context,
            metadata,
        )


def _smooth_quantile(scores: np.ndarray, alpha: float, beta: float = 15.0) -> float:
    """Differentiable smooth quantile for stable FD gradient computation."""
    s_max = float(scores.max())
    lse = float(np.log(np.mean(np.exp(beta * (scores - s_max)))) / beta) + s_max
    return (1.0 - alpha) * lse + alpha * float(scores.mean())


def calibrate_biased_template(
    W: np.ndarray, b: np.ndarray, residuals: np.ndarray, alpha: float
) -> tuple[np.ndarray, float]:
    """Conformal quantile for max-affine score with per-halfspace biases.

    Score: s(r) = max_k (w_k^T r + b_k).  Level set: {r: w_k^T r <= q - b_k}.
    """
    scores = np.max(residuals @ W.T + b[None, :], axis=1)
    q = conformal_quantile(scores, alpha)
    return q * np.ones(W.shape[0]) - b, q


def _wc_cost_at_center(
    verts: np.ndarray, center: np.ndarray, cost: np.ndarray, loss_spec: LossSpec,
) -> float:
    """min_{z=(t,1-t)} max_{y in center+verts} loss_spec.loss(y, cost, z).

    Mirrors the newsvendor/linear/quadratic dispatch in
    ``robust.solve_robust_decision`` so the fast proxy objective used inside
    gradient-based ``Ours`` variants matches the true task loss for every
    benchmark, not only the linear and quadratic-regularized ones.
    """
    if loss_spec.name == "newsvendor":
        return newsvendor_robust_via_support(
            lambda a: float(np.max(verts @ a)),
            lambda a: verts[int(np.argmax(verts @ a))],
            center,
            cost,
            loss_spec,
        ).objective

    base = cost + center
    slopes = verts[:, 0] - verts[:, 1]
    ints = verts[:, 1]
    candidates = [0.0, 1.0]
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = slopes[i] - slopes[j]
            if abs(d) > 1e-12:
                t = (ints[j] - ints[i]) / d
                if -1e-10 <= t <= 1.0 + 1e-10:
                    candidates.append(float(np.clip(t, 0.0, 1.0)))
    if loss_spec.quadratic_weight > 0.0:
        lo, hi = 0.0, 1.0
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)

        def eval_t(t: float) -> float:
            z = np.array([t, 1.0 - t])
            return float(base @ z + np.max(verts @ z) + loss_spec.penalty(z))

        fc = eval_t(c)
        fd = eval_t(d)
        for _ in range(60):
            if fc <= fd:
                hi = d
                d = c
                fd = fc
                c = hi - inv_phi * (hi - lo)
                fc = eval_t(c)
            else:
                lo = c
                c = d
                fc = fd
                d = lo + inv_phi * (hi - lo)
                fd = eval_t(d)
        candidates.append(0.5 * (lo + hi))
    return min(
        float(base @ np.array([t, 1.0 - t]))
        + float(np.max(verts @ np.array([t, 1.0 - t])))
        + loss_spec.penalty(np.array([t, 1.0 - t]))
        for t in candidates
    )


def _wc_objective_smooth(
    W_norm: np.ndarray, r_learn: np.ndarray, centers: np.ndarray,
    cost: np.ndarray, alpha: float, loss_spec: LossSpec | None = None,
    b: np.ndarray | None = None,
) -> float:
    """E_x[min_{z∈Δ_2} max_{r∈C_W} (cost+center+r)^T z] with smooth calibration.

    Supports per-halfspace biases via optional ``b`` (length-K array).
    Score: s(r) = max_k (w_k^T r + b_k); level set h_k = q - b_k.
    """
    loss_spec = loss_spec or LossSpec()
    K = W_norm.shape[0]
    b_vec = np.zeros(K) if b is None else np.asarray(b, dtype=float)
    scores = np.max(r_learn @ W_norm.T + b_vec[None, :], axis=1)
    q = _smooth_quantile(scores, alpha)
    h = q * np.ones(K) - b_vec
    try:
        verts = polytope_vertices(W_norm, h)
    except ValueError:
        return 1e6
    if len(verts) < 2:
        return 1e6
    total = sum(_wc_cost_at_center(verts, center, cost, loss_spec) for center in centers)
    return total / max(1, len(centers))


def _logsumexp_scores(R: np.ndarray, W: np.ndarray, b: np.ndarray, beta: float = 25.0) -> np.ndarray:
    """Smooth max-affine scores: logsumexp_k(w_k^T r_i + b_k) / beta."""
    raw = R @ W.T + b[None, :]
    row_max = raw.max(axis=1, keepdims=True)
    return row_max[:, 0] + np.log(np.mean(np.exp(beta * (raw - row_max)), axis=1)) / beta


def _smooth_pinball_loss(values: np.ndarray, tau: float, beta: float = 25.0) -> float:
    """Mean smoothed pinball rho_tau(u) = tau*softplus(u) + (1-tau)*softplus(-u)."""
    bv = beta * np.asarray(values, dtype=float)
    pos = np.logaddexp(0.0, bv) / beta
    neg = np.logaddexp(0.0, -bv) / beta
    return float(np.mean(tau * pos + (1.0 - tau) * neg))


def _wc_objective_fixed_radius(
    W: np.ndarray, radius: float, centers: np.ndarray,
    cost: np.ndarray, loss_spec: LossSpec,
) -> float:
    """min_{z in Delta_2} max_{Wr<=radius} g(z,y) averaged over centers.

    Uses the same vertex enumeration + golden-section as _wc_objective_smooth,
    but takes a fixed radius instead of computing a quantile from data.
    """
    K = W.shape[0]
    h = radius * np.ones(K)
    try:
        verts = polytope_vertices(W, h)
    except ValueError:
        return 1e6
    if len(verts) < 2:
        return 1e6
    total = sum(_wc_cost_at_center(verts, center, cost, loss_spec) for center in centers)
    return total / max(1, len(centers))


class YehGDAdapter(BaseMethod):
    """Decision-aware GD on free unit-norm halfspace normals W.

    Score s(r) = max_k w_k^T r (max-affine, simplified FICNN with zero bias).
    Normals are kept unit-norm to guarantee a bounded polytope.

    Key architectural limitation: unit-norm normals + uniform threshold h = q·1_K
    restrict the set to isotropic configurations (equivalent to rho=1 only).
    Our CCG grid uses non-unit-norm W via the rho stretching, which provides
    non-uniform effective thresholds and strictly contains this family.  GD also
    initializes at directions(K) (the identity-box / conformal-box configuration),
    which is a local minimum of the smooth WC objective; only 0.0008 improvement
    is observed over 200 steps because the gradient is near-zero at that point.
    """

    name = "Yeh et al. (GD, decision-aware)"
    color = "#17becf"

    def __init__(
        self,
        n_iter: int = 200,
        eta0: float = 0.08,
        decay: float = 0.99,
        fd_h: float = 1e-3,
    ) -> None:
        self.n_iter = n_iter
        self.eta0 = eta0
        self.decay = decay
        self.fd_h = fd_h

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        K = context.k_poly
        # Evenly-spaced init: guarantees positive spanning of R^2 (bounded polytope).
        # Also happens to be the identity-box / conformal-box configuration, which is
        # a local minimum of the smooth WC objective — GD barely moves from here.
        W = directions(K).copy().astype(float)

        centers = _selection_centers(context)
        cost = _cost(context)
        loss_spec = _loss(context)
        alpha = context.alpha

        def _normalize(W_raw: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(W_raw, axis=1, keepdims=True)
            return W_raw / np.maximum(norms, 1e-8)

        def obj(W_raw: np.ndarray) -> float:
            return _wc_objective_smooth(_normalize(W_raw), r_learn, centers, cost, alpha, loss_spec)

        for step in range(self.n_iter):
            eta = self.eta0 * (self.decay ** step)
            grad = np.zeros_like(W)
            for i in range(K):
                for j in range(2):
                    Wp = W.copy(); Wp[i, j] += self.fd_h
                    Wm = W.copy(); Wm[i, j] -= self.fd_h
                    grad[i, j] = (obj(Wp) - obj(Wm)) / (2.0 * self.fd_h)
            W -= eta * grad

        self.W_ = _normalize(W)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.W_,
            r_recal,
            context,
            {"selection": "free unit-norm normals GD, decision-aware"},
        )


class OursGDAdapter(BaseMethod):
    """Our method: gradient descent on (phi, log_rho) of structured rotated-box template.

    Optimizes the same decision-aware objective as the CCG grid search but via
    continuous gradient descent on the 2D parameter space (phi, log_rho), with
    multiple random restarts to escape local minima.  Final uncertainty set is
    recalibrated on an independent split, so the formal coverage guarantee applies.
    """

    name = "Ours (GD, structured)"
    color = "#e377c2"

    def __init__(
        self,
        n_init: int = 5,
        n_iter: int = 150,
        eta0: float = 0.08,
        decay: float = 0.99,
        fd_h: float = 1e-3,
    ) -> None:
        self.n_init = n_init
        self.n_iter = n_iter
        self.eta0 = eta0
        self.decay = decay
        self.fd_h = fd_h

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        centers = _selection_centers(context)
        cost = _cost(context)
        loss_spec = _loss(context)
        alpha = context.alpha
        K = context.k_poly
        rng = np.random.default_rng(context.seed + 42)

        def obj(phi: float, log_rho: float) -> float:
            W = stretched_direction_normals(K, phi, float(np.exp(log_rho)))
            return _wc_objective_smooth(W, r_learn, centers, cost, alpha, loss_spec)

        best_obj = np.inf
        best_phi, best_log_rho = 0.0, 0.0
        for _ in range(self.n_init):
            phi = float(rng.uniform(0.0, np.pi))
            log_rho = float(rng.uniform(np.log(0.55), np.log(1.8)))
            for step in range(self.n_iter):
                eta = self.eta0 * (self.decay ** step)
                g_phi = (obj(phi + self.fd_h, log_rho) - obj(phi - self.fd_h, log_rho)) / (2.0 * self.fd_h)
                g_lr = (obj(phi, log_rho + self.fd_h) - obj(phi, log_rho - self.fd_h)) / (2.0 * self.fd_h)
                phi -= eta * g_phi
                log_rho -= eta * g_lr
                phi = phi % np.pi
            curr = obj(phi, log_rho)
            if curr < best_obj:
                best_obj, best_phi, best_log_rho = curr, phi, log_rho

        self.W_ = stretched_direction_normals(K, best_phi, float(np.exp(best_log_rho)))
        self.best_phi_ = best_phi
        self.best_rho_ = float(np.exp(best_log_rho))
        self.learn_robust_objective_ = best_obj

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.W_,
            r_recal,
            context,
            {
                "selection": "structured GD with final recalibration",
                "selected_phi": self.best_phi_,
                "selected_rho": self.best_rho_,
                "learn_robust_objective": self.learn_robust_objective_,
            },
        )


class OursGDFreeAdapter(BaseMethod):
    """Our method: GD over a free 2×2 linear map A (general parallelogram polytope).

    Halfspace normals: W = directions(K) @ A^T for a free invertible A ∈ R^{2×2}.
    This is strictly more expressive than OursGDAdapter, which restricts A to
    diag(1/ρ, ρ)·R(φ)^T (2 DoF).  With a free A (4 DoF) the set can be any
    centrally-symmetric K-gon — not just axis-aligned rotated rectangles.

    Score: s(r) = max_k w_k^T r.  Calibrated on independent recal split.
    """

    name = "Ours (GD, free)"
    color = "#fd8d3c"

    def __init__(
        self,
        n_init: int = 5,
        n_iter: int = 150,
        eta0: float = 0.05,
        decay: float = 0.99,
        fd_h: float = 1e-3,
    ) -> None:
        self.n_init = n_init
        self.n_iter = n_iter
        self.eta0 = eta0
        self.decay = decay
        self.fd_h = fd_h

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        _ensure_2d(context)
        K = context.k_poly
        dirs = directions(K)
        centers = _selection_centers(context)
        cost = _cost(context)
        loss_spec = _loss(context)
        alpha = context.alpha
        rng = np.random.default_rng(context.seed + 99)

        def obj(A: np.ndarray) -> float:
            if abs(np.linalg.det(A)) < 1e-6:
                return 1e6
            return _wc_objective_smooth(dirs @ A.T, r_learn, centers, cost, alpha, loss_spec)

        best_obj = np.inf
        best_A = np.eye(2)

        for _ in range(self.n_init):
            # Initialize near a random (phi, rho) point so each restart explores a
            # different region of the 4-DoF space, but starts from a valid polytope.
            phi = float(rng.uniform(0.0, np.pi))
            rho = float(np.exp(rng.uniform(np.log(0.55), np.log(1.8))))
            rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
            A = np.diag([1.0 / rho, rho]) @ rot.T + rng.standard_normal((2, 2)) * 0.05

            for step in range(self.n_iter):
                eta = self.eta0 * (self.decay ** step)
                grad = np.zeros((2, 2))
                for i in range(2):
                    for j in range(2):
                        Ap = A.copy(); Ap[i, j] += self.fd_h
                        Am = A.copy(); Am[i, j] -= self.fd_h
                        grad[i, j] = (obj(Ap) - obj(Am)) / (2.0 * self.fd_h)
                A -= eta * grad

            curr = obj(A)
            if curr < best_obj:
                best_obj = curr
                best_A = A.copy()

        self.A_ = best_A
        self.W_ = dirs @ best_A.T
        self.learn_robust_objective_ = best_obj

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        return _set_from_template(
            self.name,
            self.color,
            self.W_,
            r_recal,
            context,
            {
                "selection": "free 2×2 linear map GD with final recalibration",
                "A": self.A_.tolist(),
                "learn_robust_objective": self.learn_robust_objective_,
            },
        )


class OursPinballGDAdapter(BaseMethod):
    """Our method: pinball-loss surrogate + gradient descent on (phi, log_rho, log_radius).

    Replaces the exact big-M conformal radius constraint with the pinball variational
    surrogate.  For fixed theta = (W, b=0), the empirical tau-quantile of
    {S_i(theta)} is characterized by minimizing mean_i rho_tau(S_i - r) over r.
    We therefore jointly minimize over (phi, log_rho, log_radius):

        min_{phi,rho,r}  min_{z in Delta_2} max_{W r_i <= r} g(z,y)
                         + lambda * mean_i rho_tau(S_i(theta) - r)

    During training, max_k(.) is smoothed with logsumexp and the pinball hinges
    with softplus.  After shape selection, the exact conformal quantile on the
    independent recal split gives the formal coverage guarantee.
    """

    name = "Ours (Pinball-GD)"
    color = "#7b3294"

    def __init__(
        self,
        n_init: int = 5,
        n_iter: int = 80,
        eta0: float = 0.08,
        decay: float = 0.97,
        fd_h: float = 1e-3,
        pinball_weight: float = 4.0,
        score_beta: float = 25.0,
        hinge_beta: float = 25.0,
    ) -> None:
        self.n_init = n_init
        self.n_iter = n_iter
        self.eta0 = eta0
        self.decay = decay
        self.fd_h = fd_h
        self.pinball_weight = pinball_weight
        self.score_beta = score_beta
        self.hinge_beta = hinge_beta

    def _surrogate(
        self,
        phi: float, log_rho: float, log_radius: float,
        r_learn: np.ndarray, centers: np.ndarray,
        cost: np.ndarray, alpha: float, K: int, loss_spec: LossSpec,
    ) -> float:
        """Evaluate the smooth pinball surrogate at (phi, log_rho, log_radius)."""
        rho = float(np.exp(log_rho))
        radius = float(np.exp(log_radius))
        W = stretched_direction_normals(K, phi, rho)
        b = np.zeros(K)
        n = len(r_learn)
        tau = min(int(np.ceil((n + 1) * (1.0 - alpha))), n) / n
        smooth_scores = _logsumexp_scores(r_learn, W, b, beta=self.score_beta)
        pinball = _smooth_pinball_loss(smooth_scores - radius, tau, beta=self.hinge_beta)
        robust_obj = _wc_objective_fixed_radius(W, radius, centers, cost, loss_spec)
        if robust_obj >= 1e5:
            return 1e6
        return robust_obj + self.pinball_weight * pinball

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        if context.y_dim != 2:
            # The (phi, rho) parameterization is 2D-specific. For the
            # dimension-sensitivity sweep, fall back to selecting among a
            # paired-random positively-spanning library in arbitrary
            # dimension, matching the fallback used elsewhere for d>2.
            selected = _select_min_robust(
                _paired_random_cut_templates(context, n_templates=max(context.n_croms, 24)),
                r_learn,
                context,
            )
            self.W_ = selected["W"]
            self.best_phi_ = float("nan")
            self.best_rho_ = float("nan")
            self.best_radius_ = float("nan")
            self.surrogate_obj_ = selected["learn_robust_objective"]
            self._fallback_dim_ = True
            return
        self._fallback_dim_ = False
        K = context.k_poly
        centers = _selection_centers(context)
        cost = _cost(context)
        loss_spec = _loss(context)
        alpha = context.alpha
        rng = np.random.default_rng(context.seed + 311)

        def obj(phi: float, log_rho: float, log_radius: float) -> float:
            return self._surrogate(phi, log_rho, log_radius, r_learn, centers, cost, alpha, K, loss_spec)

        best_obj = np.inf
        best_phi, best_log_rho, best_log_radius = 0.0, 0.0, 0.0

        init_phis = [np.deg2rad(25.0)] + list(rng.uniform(0.0, np.pi, max(0, self.n_init - 1)))
        for start_id, phi0 in enumerate(init_phis):
            phi = float(phi0)
            log_rho = 0.0 if start_id == 0 else float(rng.uniform(np.log(0.55), np.log(1.8)))
            W0 = stretched_direction_normals(K, phi, float(np.exp(log_rho)))
            q0 = conformal_quantile(np.max(r_learn @ W0.T, axis=1), alpha)
            log_radius = float(np.log(np.clip(q0, 0.03, 5.0)))

            for step in range(self.n_iter):
                eta = self.eta0 * (self.decay ** step)
                x = np.array([phi, log_rho, log_radius])
                grad = np.zeros(3)
                for j in range(3):
                    xp = x.copy(); xp[j] += self.fd_h
                    xm = x.copy(); xm[j] -= self.fd_h
                    fp = obj(float(xp[0]), float(xp[1]), float(xp[2]))
                    fm = obj(float(xm[0]), float(xm[1]), float(xm[2]))
                    grad[j] = (fp - fm) / (2.0 * self.fd_h)
                phi = float((phi - eta * grad[0]) % np.pi)
                log_rho = float(np.clip(log_rho - eta * grad[1], np.log(0.40), np.log(2.50)))
                log_radius = float(np.clip(log_radius - eta * grad[2], np.log(0.02), np.log(5.0)))
                if np.linalg.norm(grad) < 1e-4:
                    break

            curr = obj(phi, log_rho, log_radius)
            if curr < best_obj:
                best_obj = curr
                best_phi, best_log_rho, best_log_radius = phi, log_rho, log_radius

        self.W_ = stretched_direction_normals(K, best_phi, float(np.exp(best_log_rho)))
        self.best_phi_ = best_phi
        self.best_rho_ = float(np.exp(best_log_rho))
        self.best_radius_ = float(np.exp(best_log_radius))
        self.surrogate_obj_ = best_obj

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        selection = (
            "paired-random K-cut library fallback (y_dim != 2)"
            if getattr(self, "_fallback_dim_", False)
            else "pinball-loss GD on (phi, log_rho, log_radius)"
        )
        return _set_from_template(
            self.name,
            self.color,
            self.W_,
            r_recal,
            context,
            {
                "selection": selection,
                "selected_phi": self.best_phi_,
                "selected_rho": self.best_rho_,
                "train_radius": self.best_radius_,
                "surrogate_obj": self.surrogate_obj_,
            },
        )


class OursCCGFreeAdapter(BaseMethod):
    """Ours (CCG free): exact MIQCP over free (W, b) — Proposition 1, linear g.

    Solves eq. (theta-joint-linear) with unrestricted (W, b) via Gurobi NonConvex=2.
    Joint variables: K×2 normal matrix W, K biases b, decision z ∈ Δ₂,
    dual vars λ ≥ 0, conformal threshold q, binary coverage indicators t ∈ {0,1}^n.

    Bilinear coupling W^T λ = z (dual balance) and bilinear objective terms
    q·1_K^T λ, b^T λ are handled by Gurobi's bilinear MIQP solver.

    Coverage constraint w_k^T r_i + b_k ≤ q + M(1-t_i) is linear in (W,b,q,t)
    because residuals r_i are observed data.

    Requires: gurobipy (Gurobi ≥ 10.0, license required).
    """

    name = "Ours (CCG free)"
    color = "#17becf"

    def __init__(
        self,
        K: int | None = None,
        n_iter: int = 4,
        big_M_factor: float = 10.0,
        time_limit: float = 300.0,
    ) -> None:
        self.K = K
        self.n_iter = n_iter
        self.big_M_factor = big_M_factor
        self.time_limit = time_limit

    @staticmethod
    def _lp_step(
        W: np.ndarray, b: np.ndarray, q: float,
        center: np.ndarray, cost: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fix (W,b,q): LP over (z,λ).  min (cost+center)^T z + (q−b)^T λ
        s.t. W^T λ = z, 1^T z = 1, z ≥ 0, λ ≥ 0."""
        from scipy.optimize import linprog

        K, d = W.shape
        h = q * np.ones(K) - b
        c = np.concatenate([cost + center, h])
        A_eq = np.block([[np.eye(d), -W.T], [np.ones((1, d)), np.zeros((1, K))]])
        b_eq = np.concatenate([np.zeros(d), [1.0]])
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * (d + K), method="highs")
        if res.status != 0:
            return np.ones(d) / d, np.zeros(K), float("nan")
        return res.x[:d], res.x[d:], float(res.fun)

    @staticmethod
    def _milp_step(
        r_learn: np.ndarray, z_fix: np.ndarray, lam_fix: np.ndarray,
        kappa: int, K: int, W_bnd: float, big_M: float, eps_W: float,
        u: np.ndarray, u_next: np.ndarray, time_limit: float,
        gp: object, GRB: object,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fix (z,λ): MILP over (W,b,q,t).  Bilinear W^T λ = z is linear since λ fixed.
        Objective: q·sum(λ) − b^T λ  (linear in q,b with fixed λ ≥ 0)."""
        n, d = r_learn.shape
        m = gp.Model()
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = time_limit
        m.Params.MIPGap = 0.005

        W_v = m.addVars(K, d, lb=-W_bnd, ub=W_bnd, name="W")
        b_v = m.addVars(K, lb=-W_bnd, ub=0.0, name="b")  # b_k ≤ 0 → h_k = q−b_k ≥ 0
        q_v = m.addVar(lb=0.0, ub=W_bnd, name="q")
        t_v = m.addVars(n, vtype=GRB.BINARY, name="t")

        m.addConstr(gp.quicksum(t_v[i] for i in range(n)) >= kappa)

        for k in range(K):
            m.addConstr(float(u[k, 0]) * W_v[k, 1] - float(u[k, 1]) * W_v[k, 0] >= eps_W)
            m.addConstr(float(u_next[k, 0]) * W_v[k, 1] - float(u_next[k, 1]) * W_v[k, 0] <= -eps_W)

        # Dual balance is linear when λ is fixed: sum_k W[k,j]*lam_fix[k] = z_fix[j]
        for j in range(d):
            m.addConstr(
                gp.quicksum(W_v[k, j] * float(lam_fix[k]) for k in range(K)) == float(z_fix[j])
            )

        for i in range(n):
            for k in range(K):
                lhs = gp.quicksum(W_v[k, j] * float(r_learn[i, j]) for j in range(d)) + b_v[k]
                m.addConstr(lhs <= q_v + big_M * (1 - t_v[i]))

        lam_sum = float(np.sum(lam_fix))
        obj = q_v * lam_sum + gp.quicksum(-float(lam_fix[k]) * b_v[k] for k in range(K))
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.SolCount == 0:
            raise RuntimeError("MILP step found no feasible solution.")
        W_sol = np.array([[W_v[k, j].X for j in range(d)] for k in range(K)])
        b_sol = np.array([b_v[k].X for k in range(K)])
        return W_sol, b_sol, float(q_v.X)

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise NotImplementedError("OursCCGFreeAdapter requires gurobipy (Gurobi).")

        _ensure_2d(context)
        n, d = r_learn.shape
        K = self.K if self.K is not None else context.k_poly
        center = np.asarray(context.robust_center, dtype=float)
        cost = _cost(context)
        alpha = context.alpha
        kappa = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)

        # Sector geometry: restrict normal k to sector [2πk/K, 2π(k+1)/K].
        # Strict inequalities (eps_W > 0) prevent normals from coinciding on boundaries,
        # which would make the polytope unbounded.
        angles = np.linspace(0.0, 2.0 * np.pi, K, endpoint=False)
        u = np.column_stack([np.cos(angles), np.sin(angles)])
        u_next = np.roll(u, -1, axis=0)

        # Initialize from symmetric directions(K) with b=0.
        D_init = directions(K)
        scores_init = np.max(r_learn @ D_init.T, axis=1)
        q_cur = float(conformal_quantile(scores_init, alpha))
        W_cur, b_cur = D_init.copy(), np.zeros(K)

        eps_W = 0.15 * q_cur
        W_bnd = self.big_M_factor * q_cur
        r_max = float(np.max(np.linalg.norm(r_learn, axis=1)))
        big_M = float(np.sqrt(d)) * W_bnd * r_max + 2.0 * W_bnd
        per_step_time = self.time_limit / max(self.n_iter, 1)

        best_obj, best_W, best_b = np.inf, W_cur.copy(), b_cur.copy()

        # Alternating LP ↔ MILP coordinate descent.
        # Each step is a standard LP or MILP — no bilinear variables in any single step.
        for _ in range(self.n_iter):
            z_opt, lam_opt, obj_lp = self._lp_step(W_cur, b_cur, q_cur, center, cost)
            if not np.isnan(obj_lp) and obj_lp < best_obj:
                best_obj, best_W, best_b = obj_lp, W_cur.copy(), b_cur.copy()
            if np.max(lam_opt) < 1e-10:
                break
            try:
                W_cur, b_cur, q_cur = self._milp_step(
                    r_learn, z_opt, lam_opt, kappa, K, W_bnd, big_M,
                    eps_W, u, u_next, per_step_time, gp, GRB,
                )
            except RuntimeError:
                break

        self.W_ = best_W
        self.b_ = best_b
        self.fit_obj_ = best_obj

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        # Fix learned (W, b) and compute conformal quantile on the held-out recal split.
        scores = np.max(r_recal @ self.W_.T + self.b_[None, :], axis=1)
        q = float(conformal_quantile(scores, context.alpha))
        h = q * np.ones(self.W_.shape[0]) - self.b_
        return SetResult(
            self.name,
            kind="polytope",
            color=self.color,
            W=self.W_,
            h=h,
            metadata={"q": q, "fit_obj": self.fit_obj_},
        )


class OursGurobiMIQCPAdapter(BaseMethod):
    """Exact MIQCP over the common SPD-paired class, fixed learning context.

    Reuses the compact, positively-spanning class from the controlled
    exact-versus-pinball diagnostic (``run_exact_vs_pinball.py``):
    ``W=(a_1,a_2,-a_1,-a_2)^T`` with ``A=[a_1;a_2]=A^T``, ``||W||_F=1``,
    ``lambda_min(A)>=eig_lb``, and ``b=0``.  The objective is evaluated at one
    representative learning context (the mean predicted center over the
    learning split, ``context.robust_center``), matching that diagnostic's
    fixed-context protocol.  This is a compact, numerically well-posed
    subclass of the free ``(W,b)`` formulation: an earlier version of this
    adapter optimized over unrestricted ``(W,b)`` bounds and produced
    numerically degenerate (near-unbounded or near-zero-volume) polytopes
    under a short time limit at ``n=300``.  Only registered for the synthetic
    linear task because the formulation is for linear loss.
    """

    name = "Ours (Gurobi MIQCP)"
    color = "crimson"
    supported_experiments = ("synthetic_linear",)

    def __init__(self, time_limit: float = 60.0, eig_lb: float = 0.15) -> None:
        self.time_limit = time_limit
        self.eig_lb = eig_lb

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise NotImplementedError("OursGurobiMIQCPAdapter requires gurobipy (Gurobi).")

        _ensure_2d(context)
        loss_spec = _loss(context)
        if loss_spec.name != "linear" or loss_spec.quadratic_weight != 0.0:
            raise ValueError("Ours (Gurobi MIQCP) is implemented for linear loss only.")
        if context.k_poly != 4:
            raise ValueError("Ours (Gurobi MIQCP) uses a fixed K=4 paired-normal class.")

        r_learn = np.asarray(r_learn, dtype=float)
        n, d = r_learn.shape
        center = np.asarray(context.robust_center, dtype=float)
        cost = _cost(context)
        alpha = context.alpha
        kappa = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
        eig_lb = self.eig_lb

        residual_bound = float(np.max(np.linalg.norm(r_learn, axis=1)))
        q_ub = residual_bound / np.sqrt(2.0) + 1e-8
        big_m = q_ub
        dual_ub = 1.0 / eig_lb

        model = gp.Model("ours_gurobi_miqcp_spd_paired")
        model.setParam("OutputFlag", 0)
        model.setParam("NonConvex", 2)
        model.setParam("TimeLimit", float(self.time_limit))
        model.setParam("MIPGap", 5e-3)
        model.setParam("MIPGapAbs", 1e-5)
        model.setParam("Threads", 1)

        a00 = model.addVar(lb=eig_lb, ub=1.0 / np.sqrt(2.0), name="a00")
        a01 = model.addVar(lb=-0.5, ub=0.5, name="a01")
        a11 = model.addVar(lb=eig_lb, ub=1.0 / np.sqrt(2.0), name="a11")
        q_v = model.addVar(lb=0.0, ub=q_ub, name="q")
        t_v = model.addVars(n, vtype=GRB.BINARY, name="t")
        z_v = model.addVars(2, lb=0.0, ub=1.0, name="z")
        lam_v = model.addVars(4, lb=0.0, ub=dual_ub, name="lam")

        model.addQConstr(
            2.0 * (a00 * a00 + 2.0 * a01 * a01 + a11 * a11) == 1.0,
            name="frobenius_normalization",
        )
        model.addQConstr((a00 - eig_lb) * (a11 - eig_lb) >= a01 * a01, name="interior_margin")

        model.addQConstr(
            a00 * (lam_v[0] - lam_v[2]) + a01 * (lam_v[1] - lam_v[3]) == z_v[0],
            name="dual_balance_0",
        )
        model.addQConstr(
            a01 * (lam_v[0] - lam_v[2]) + a11 * (lam_v[1] - lam_v[3]) == z_v[1],
            name="dual_balance_1",
        )
        model.addConstr(z_v[0] + z_v[1] == 1.0, name="simplex")
        model.addConstr(gp.quicksum(t_v[i] for i in range(n)) >= kappa, name="kappa")

        for i, eps in enumerate(r_learn):
            v0 = float(eps[0]) * a00 + float(eps[1]) * a01
            v1 = float(eps[0]) * a01 + float(eps[1]) * a11
            rhs = q_v + big_m * (1 - t_v[i])
            model.addConstr(v0 <= rhs, name=f"score_pos0_{i}")
            model.addConstr(v1 <= rhs, name=f"score_pos1_{i}")
            model.addConstr(-v0 <= rhs, name=f"score_neg0_{i}")
            model.addConstr(-v1 <= rhs, name=f"score_neg1_{i}")

        base = cost + center
        obj = float(base[0]) * z_v[0] + float(base[1]) * z_v[1]
        obj += q_v * gp.quicksum(lam_v[k] for k in range(4))
        model.setObjective(obj, GRB.MINIMIZE)

        a00.Start = 0.5
        a01.Start = 0.0
        a11.Start = 0.5
        start_A = np.eye(2) * 0.5
        start_W = np.vstack([start_A[0], start_A[1], -start_A[0], -start_A[1]])
        start_scores = np.max(r_learn @ start_W.T, axis=1)
        start_q = float(conformal_quantile(start_scores, alpha))
        q_v.Start = start_q
        for i in range(n):
            t_v[i].Start = float(start_scores[i] <= start_q + 1e-10)

        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError(f"Gurobi MIQCP found no feasible solution: status={model.Status}")

        A = np.array([[a00.X, a01.X], [a01.X, a11.X]])
        self.W_ = np.vstack([A[0], A[1], -A[0], -A[1]])
        self.b_ = np.zeros(4)
        self.q_train_ = float(q_v.X)
        self.z_train_ = np.array([z_v[0].X, z_v[1].X])
        self.fit_obj_ = float(model.ObjVal)
        self.status_ = int(model.Status)
        self.runtime_ = float(model.Runtime)
        self.mip_gap_ = float(model.MIPGap) if model.SolCount > 0 else float("nan")

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        scores = np.max(r_recal @ self.W_.T + self.b_[None, :], axis=1)
        q = float(conformal_quantile(scores, context.alpha))
        h = q * np.ones(self.W_.shape[0]) - self.b_
        return SetResult(
            self.name,
            kind="polytope",
            color=self.color,
            W=self.W_,
            h=h,
            metadata={
                "q": q,
                "q_train": self.q_train_,
                "z_train": self.z_train_.tolist(),
                "fit_obj": self.fit_obj_,
                "status": self.status_,
                "runtime": self.runtime_,
                "mip_gap": self.mip_gap_,
                "time_limit": self.time_limit,
            },
        )


# ---------------------------------------------------------------------------
# Ours
#
# Every learning problem is first attempted with the exact MIQCP of
# Corollary 1 (linear loss; the quadratic-regularized loss only adds the convex
# term g_0(z)), solved by Gurobi under a wall-clock limit of
# ``context.miqcp_time_limit`` seconds (30 minutes by default).  The MIQCP
# averages the robust loss over the learning contexts
# (``context.selection_centers``), i.e. it targets the empirical objective that
# defines theta_hat(D_n), and is posed over the compact paired-normal class
# W = (a_1, a_2, -a_1, -a_2)^T, A = [a_1; a_2] = A^T, 2||A||_F^2 = 1,
# lambda_min(A) >= eta, b = 0 (the same class as the exact-versus-pinball
# study).  The geometry is used only if Gurobi certifies optimality (relative
# gap <= 0.5% or absolute gap <= 1e-3).  Otherwise -- and whenever the MIQCP
# does not apply (newsvendor loss, K != 4, d != 2) -- we use the pinball-loss
# gradient method: Pinball-GD over rotated/stretched K-gons for d = 2, and the
# stochastic pinball-subgradient method (Alg. 2) over linear images of a fixed
# positively spanning polytope for d > 2.  No variant searches a fixed library.
# ---------------------------------------------------------------------------

_MIQCP_MIP_GAP = 5e-3
_MIQCP_MIP_GAP_ABS = 1e-3
_OURS_GEOMETRY_CACHE: dict[tuple, dict] = {}


def _array_digest(arr: np.ndarray) -> str:
    import hashlib

    a = np.ascontiguousarray(np.asarray(arr, dtype=float))
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


def _solve_contextual_miqcp(
    r_learn: np.ndarray,
    centers: np.ndarray,
    cost: np.ndarray,
    alpha: float,
    quadratic_weight: float = 0.0,
    eig_lb: float = 0.15,
    time_limit: float = 1800.0,
    threads: int = 1,
) -> dict:
    """Exact contextual MIQCP (Corollary 1) over the SPD-paired K=4 class.

    min (1/N) sum_j [(c + f_j)^T z_j + beta (z_j0 - z_j1)^2 + 1^T mu_j]
    s.t. A (mu_j^+ - mu_j^-) = q z_j, z_j in simplex, mu_j >= 0   (mu_j = q * lambda_j)
         |A eps_i|_inf <= q + M (1 - t_i), sum_i t_i >= ceil((n+1)(1-alpha))
         2 ||A||_F^2 = 1, (a00 - eta)(a11 - eta) >= a01^2, a00, a11 >= eta.
    """
    import gurobipy as gp
    from gurobipy import GRB

    r_learn = np.asarray(r_learn, dtype=float)
    centers = np.asarray(centers, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n, N = len(r_learn), len(centers)
    kappa = min(int(np.ceil((n + 1) * (1.0 - alpha))), n)
    q_ub = float(np.max(np.linalg.norm(r_learn, axis=1))) / np.sqrt(2.0) + 1e-8
    big_m = q_ub
    mu_ub = q_ub / eig_lb

    model = gp.Model("ours_contextual_miqcp")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = _MIQCP_MIP_GAP
    model.Params.MIPGapAbs = _MIQCP_MIP_GAP_ABS
    model.Params.Threads = int(threads)

    a00 = model.addVar(lb=eig_lb, ub=1.0 / np.sqrt(2.0), name="a00")
    a01 = model.addVar(lb=-0.5, ub=0.5, name="a01")
    a11 = model.addVar(lb=eig_lb, ub=1.0 / np.sqrt(2.0), name="a11")
    q_v = model.addVar(lb=0.0, ub=q_ub, name="q")
    t_v = model.addVars(n, vtype=GRB.BINARY, name="t")
    z_v = model.addVars(N, 2, lb=0.0, ub=1.0, name="z")
    mu_v = model.addVars(N, 4, lb=0.0, ub=mu_ub, name="mu")

    model.addQConstr(2.0 * (a00 * a00 + 2.0 * a01 * a01 + a11 * a11) == 1.0, name="frobenius_normalization")
    model.addQConstr((a00 - eig_lb) * (a11 - eig_lb) >= a01 * a01, name="interior_margin")
    for j in range(N):
        u0 = mu_v[j, 0] - mu_v[j, 2]
        u1 = mu_v[j, 1] - mu_v[j, 3]
        model.addQConstr(a00 * u0 + a01 * u1 == q_v * z_v[j, 0], name=f"dual_balance_0_{j}")
        model.addQConstr(a01 * u0 + a11 * u1 == q_v * z_v[j, 1], name=f"dual_balance_1_{j}")
        model.addConstr(z_v[j, 0] + z_v[j, 1] == 1.0, name=f"simplex_{j}")
    model.addConstr(gp.quicksum(t_v[i] for i in range(n)) >= kappa, name="kappa")
    for i, eps in enumerate(r_learn):
        v0 = float(eps[0]) * a00 + float(eps[1]) * a01
        v1 = float(eps[0]) * a01 + float(eps[1]) * a11
        rhs = q_v + big_m * (1 - t_v[i])
        model.addConstr(v0 <= rhs)
        model.addConstr(-v0 <= rhs)
        model.addConstr(v1 <= rhs)
        model.addConstr(-v1 <= rhs)

    obj = gp.QuadExpr()
    for j in range(N):
        base = cost + centers[j]
        obj += float(base[0]) * z_v[j, 0] + float(base[1]) * z_v[j, 1]
        if quadratic_weight > 0.0:
            obj += float(quadratic_weight) * (z_v[j, 0] - z_v[j, 1]) * (z_v[j, 0] - z_v[j, 1])
        obj += gp.quicksum(mu_v[j, k] for k in range(4))
    model.setObjective(obj * (1.0 / N), GRB.MINIMIZE)

    # Warm start at the axis-aligned box A = I/2.
    a00.Start, a01.Start, a11.Start = 0.5, 0.0, 0.5
    start_scores = 0.5 * np.max(np.abs(r_learn), axis=1)
    start_q = float(conformal_quantile(start_scores, alpha))
    q_v.Start = start_q
    for i in range(n):
        t_v[i].Start = float(start_scores[i] <= start_q + 1e-12)

    model.optimize()
    out = {
        "status": int(model.Status),
        "runtime": float(model.Runtime),
        "certified": bool(model.Status == GRB.OPTIMAL),
        "n_contexts": int(N),
    }
    if model.SolCount == 0:
        out.update({"W": None, "certified": False})
        return out
    A = np.array([[a00.X, a01.X], [a01.X, a11.X]])
    out.update(
        {
            "W": np.vstack([A[0], A[1], -A[0], -A[1]]),
            "b": np.zeros(4),
            "q_train": float(q_v.X),
            "incumbent": float(model.ObjVal),
            "bound": float(model.ObjBound),
            "gap_abs": float(abs(model.ObjVal - model.ObjBound)),
            "gap_rel": float(model.MIPGap),
        }
    )
    return out


def _positively_spanning_directions(d: int, k: int) -> np.ndarray:
    """K fixed unit directions in R^d that positively span R^d (+-e_i first,
    then +-s/sqrt(d) for sign vectors s with s_1 = +1)."""
    from itertools import product

    if k < 2 * d:
        raise ValueError(f"Need K >= 2d positively spanning directions; got K={k}, d={d}.")
    dirs: list[np.ndarray] = []
    for i in range(d):
        e = np.zeros(d)
        e[i] = 1.0
        dirs.extend([e, -e])
    for tail in product([1.0, -1.0], repeat=d - 1):
        if len(dirs) >= k:
            break
        s = np.r_[1.0, np.asarray(tail)] / np.sqrt(d)
        dirs.append(s)
        if len(dirs) < k:
            dirs.append(-s)
    if len(dirs) < k:
        raise ValueError(f"Not enough sign directions for K={k}, d={d}.")
    return np.asarray(dirs[:k], dtype=float)


def _robust_lp_linear(W: np.ndarray, radius: float, center: np.ndarray, cost: np.ndarray):
    """L_x(W, r) = min_{z in simplex, lam >= 0, W^T lam = z} (cost + center)^T z + r 1^T lam.

    Returns (value, dV/dW, dV/dr) using LP sensitivity (envelope theorem)."""
    from scipy.optimize import linprog

    K, d = W.shape
    c = np.concatenate([cost + center, radius * np.ones(K)])
    A_eq = np.block([[-np.eye(d), W.T], [np.ones((1, d)), np.zeros((1, K))]])
    b_eq = np.r_[np.zeros(d), 1.0]
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * (d + K), method="highs")
    if res.status != 0:
        return np.inf, np.zeros_like(W), 0.0
    lam = res.x[d:]
    y = res.eqlin.marginals[:d]
    return float(res.fun), -np.outer(lam, y), float(lam.sum())


def _fit_pinball_subgradient_nd(
    r_learn: np.ndarray,
    context: BaselineContext,
    n_iter: int = 400,
    context_batch: int = 16,
    sample_batch: int = 64,
    step: float = 0.05,
    pinball_weight: float = 15.0,
    eval_every: int = 20,
) -> tuple[np.ndarray, dict]:
    """Stochastic pinball-subgradient method (Alg. 2) for linear loss in d dims.

    Geometry W = D M with D fixed positively spanning directions and M a learned
    d x d linear map (projected to ||M||_F = sqrt(d) and condition number <= 5);
    the radius r is learned jointly through the pinball penalty.  The iterate
    with the best exact objective (exact conformal radius on D_n, all learning
    contexts) is returned.
    """
    loss_spec = _loss(context)
    if loss_spec.name != "linear":
        raise ValueError("The d>2 pinball-subgradient method is implemented for the linear loss.")
    d, K = context.y_dim, context.k_poly
    D = _positively_spanning_directions(d, K)
    centers = _selection_centers(context)
    cost = _cost(context)
    n = len(r_learn)
    tau = min(int(np.ceil((n + 1) * (1.0 - context.alpha))), n) / n
    rng = np.random.default_rng(context.seed + 733)

    def project(M: np.ndarray, r: float) -> tuple[np.ndarray, float]:
        U, s, Vt = np.linalg.svd(M)
        s = np.clip(s, s.max() / 5.0, None)
        M = (U * s) @ Vt
        scale = np.sqrt(d) / np.linalg.norm(M)
        return M * scale, float(np.clip(r * scale, 1e-4, 1e3))

    def exact_objective(M: np.ndarray) -> float:
        W = D @ M
        r = float(conformal_quantile(np.max(r_learn @ W.T, axis=1), context.alpha))
        return float(np.mean([_robust_lp_linear(W, r, c, cost)[0] for c in centers]))

    cov = np.cov(r_learn.T) + 1e-6 * np.eye(d)
    evals, evecs = np.linalg.eigh(cov)
    starts = [np.eye(d), (evecs / np.sqrt(evals)) @ evecs.T]
    best_obj, best_M = np.inf, None
    for M0 in starts:
        M, r = project(M0, float(conformal_quantile(np.max(r_learn @ (D @ M0).T, axis=1), context.alpha)))
        for it in range(n_iter):
            eta = step / np.sqrt(it + 1.0)
            W = D @ M
            ci = rng.choice(len(centers), size=min(context_batch, len(centers)), replace=False)
            gW = np.zeros_like(W)
            gr = 0.0
            for j in ci:
                _, dW, dr = _robust_lp_linear(W, r, centers[j], cost)
                gW += dW / len(ci)
                gr += dr / len(ci)
            si = rng.choice(n, size=min(sample_batch, n), replace=False)
            raw = r_learn[si] @ W.T
            kstar = np.argmax(raw, axis=1)
            scores = raw[np.arange(len(si)), kstar]
            xi = np.where(scores > r, tau, -(1.0 - tau))
            gW_pin = np.zeros_like(W)
            np.add.at(gW_pin, kstar, xi[:, None] * r_learn[si])
            gW += pinball_weight * gW_pin / len(si)
            gr_total = gr - pinball_weight * float(np.mean(xi))
            M = M - eta * (D.T @ gW)
            r = r - (eta / pinball_weight) * gr_total
            M, r = project(M, r)
            if (it + 1) % eval_every == 0 or it == n_iter - 1:
                val = exact_objective(M)
                if val < best_obj:
                    best_obj, best_M = val, M.copy()
    return D @ best_M, {"pinball_exact_learn_objective": best_obj}


def _fit_ours_geometry(r_learn: np.ndarray, context: BaselineContext) -> tuple[dict, bool]:
    """Fit (or reuse) the Ours geometry; returns (geometry, was_cached)."""
    import time as _time

    loss_spec = _loss(context)
    centers = _selection_centers(context)
    key = (
        _array_digest(r_learn),
        _array_digest(centers),
        _array_digest(_cost(context)),
        loss_spec.name,
        float(loss_spec.quadratic_weight),
        float(context.alpha),
        int(context.k_poly),
        int(context.y_dim),
        int(context.seed),
        float(context.miqcp_time_limit),
    )
    if key in _OURS_GEOMETRY_CACHE:
        return _OURS_GEOMETRY_CACHE[key], True

    t0 = _time.perf_counter()
    geometry: dict = {"miqcp_attempted": False}
    miqcp_applies = (
        context.y_dim == 2
        and context.k_poly == 4
        and loss_spec.name in ("linear", "quadratic_regularized")
        and context.miqcp_time_limit > 0
    )
    if miqcp_applies:
        try:
            res = _solve_contextual_miqcp(
                r_learn,
                centers,
                _cost(context),
                context.alpha,
                quadratic_weight=float(loss_spec.quadratic_weight),
                time_limit=float(context.miqcp_time_limit),
            )
        except Exception as exc:
            raise RuntimeError(
                "Ours requires gurobipy and a Gurobi license for its exact MIQCP step "
                f"(pass --miqcp-time-limit 0 to use only the pinball-loss method): {exc}"
            ) from exc
        geometry.update(
            {
                "miqcp_attempted": True,
                "miqcp_certified": bool(res.get("certified", False)),
                "miqcp_status": res.get("status"),
                "miqcp_runtime": res.get("runtime"),
                "miqcp_gap_rel": res.get("gap_rel"),
                "miqcp_gap_abs": res.get("gap_abs"),
                "miqcp_incumbent": res.get("incumbent"),
                "miqcp_bound": res.get("bound"),
                "miqcp_n_contexts": res.get("n_contexts"),
                "miqcp_time_limit": float(context.miqcp_time_limit),
            }
        )
        if res.get("certified") and res.get("W") is not None:
            geometry.update({"W": res["W"], "b": res["b"], "solver": "miqcp"})

    if "W" not in geometry:
        if context.y_dim == 2:
            pinball = OursPinballGDAdapter()
            pinball.fit(r_learn, context)
            geometry.update(
                {
                    "W": pinball.W_,
                    "b": np.zeros(pinball.W_.shape[0]),
                    "solver": "pinball-gd",
                    "pinball_phi": pinball.best_phi_,
                    "pinball_rho": pinball.best_rho_,
                }
            )
        else:
            W, info = _fit_pinball_subgradient_nd(r_learn, context)
            geometry.update({"W": W, "b": np.zeros(W.shape[0]), "solver": "pinball-subgradient", **info})
    geometry["fit_time"] = _time.perf_counter() - t0
    _OURS_GEOMETRY_CACHE[key] = geometry
    return geometry, False


def _ours_metadata(geometry: dict, cached: bool) -> dict:
    meta = {k: v for k, v in geometry.items() if k not in ("W", "b")}
    meta["fit_time_cached"] = float(geometry["fit_time"]) if cached else 0.0
    return meta


class OursAdapter(BaseMethod):
    """Ours: exact MIQCP when Gurobi certifies it within the time limit,
    otherwise the pinball-loss gradient method; final radius recalibrated on D_m."""

    name = "Ours (MIQCP/pinball)"
    color = "crimson"

    def fit(self, r_learn: np.ndarray, context: BaselineContext) -> None:
        self.geometry_, self.cached_ = _fit_ours_geometry(r_learn, context)

    def calibrate(self, r_recal: np.ndarray, context: BaselineContext) -> SetResult:
        W, b = self.geometry_["W"], self.geometry_["b"]
        h, q = calibrate_biased_template(W, b, r_recal, context.alpha)
        return SetResult(
            self.name,
            kind="polytope",
            color=self.color,
            W=W,
            h=h,
            metadata={
                "q": q,
                "K": W.shape[0],
                "selection": f"Ours ({self.geometry_['solver']}) with final recalibration",
                **_ours_metadata(self.geometry_, self.cached_),
            },
        )


def baseline_registry(include_unfinished: bool = False) -> list[BaseMethod]:
    ready: list[BaseMethod] = [
        BonferroniBox(),
        ConformalBox(),
        ConformalBall(),
        MahalanobisEllipseAdapter(),
        PCPAdapter(),
        FCROMSAdapter(10),
        FCROMSAdapter(20),
        PICNNAdapter(),
        YehE2EAdapter(),
        YehE2EAdapter(task_loss="robust", name="Yeh et al. (PICNN E2E, robust loss)"),
        YehConditionalAdapter(),
        DecisionAgnosticPolytope(),
        DecisionAwareNoRecal(),
        SizeAwarePolytope(),
        OursDecisionAwarePolytope(),
        OursGurobiMIQCPAdapter(),
        OursCCGFreeAdapter(),
        OursGDAdapter(),
        OursGDFreeAdapter(),
        OursPinballGDAdapter(),
        OursAdapter(),
    ]
    if not include_unfinished:
        return ready
    return ready + [FICNNAdapter()]
