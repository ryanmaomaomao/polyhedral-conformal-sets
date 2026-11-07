from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from experiments.baselines import BaselineContext, baseline_registry
from experiments.plots import plot_paper_figure, plot_residual_sets
from experiments.predictors import fit_ridge_predictor, residuals
from experiments.real_data import make_energy_splits
from experiments.robust import LossSpec, average_robust_metrics, empirical_coverage
from experiments.sets import set_area
from experiments.synthetic_xy import SyntheticSplits, make_splits


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT / "paper"
DATA_DIR = EXPERIMENT_ROOT / "data"

SCENARIO_KEYS = ["scenario", "y_dim", "k_poly", "n_train", "n_learn", "n_recal", "n_test"]
ID_KEYS = ["experiment", "dataset", "loss", "quadratic_weight", *SCENARIO_KEYS, "seed", "method"]
METRIC_KEYS = [
    "coverage",
    "volume",
    "avg_robust_objective",
    "avg_realized_loss",
    "avg_realized_regret",
    "avg_z0",
    "avg_worst_y0",
    "avg_worst_y1",
    "n_robust_eval",
    "time",
]


@dataclass(frozen=True)
class ExperimentConfig:
    key: str
    name: str
    dataset: str
    loss_spec: LossSpec
    cost: np.ndarray


def _experiment_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    cost = np.asarray(args.cost, dtype=float)
    available = {
        "synthetic-linear": ExperimentConfig(
            key="synthetic-linear",
            name="synthetic_linear",
            dataset="synthetic2d",
            loss_spec=LossSpec(name="linear"),
            cost=cost,
        ),
        "synthetic-quad": ExperimentConfig(
            key="synthetic-quad",
            name="synthetic_quadratic_regularized",
            dataset="synthetic2d",
            loss_spec=LossSpec(
                name="quadratic_regularized",
                quadratic_weight=float(args.quadratic_weight),
            ),
            cost=cost,
        ),
        "synthetic-newsvendor": ExperimentConfig(
            key="synthetic-newsvendor",
            name="synthetic_newsvendor",
            dataset="synthetic2d",
            loss_spec=LossSpec(
                name="newsvendor",
                newsvendor_c_o=(0.3, 0.3),
                newsvendor_c_u=(0.7, 0.7),
            ),
            cost=cost,
        ),
        "energy-linear": ExperimentConfig(
            key="energy-linear",
            name="energy_efficiency_linear",
            dataset="uci_energy_efficiency",
            loss_spec=LossSpec(name="linear"),
            cost=cost,
        ),
    }

    requested = [item.strip() for item in args.experiments.split(",") if item.strip()]
    if requested == ["all"]:
        return list(available.values())

    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f"Unknown experiments {unknown}. Available: {sorted(available)} or all.")
    return [available[item] for item in requested]


def _make_experiment_splits(config: ExperimentConfig, seed: int, args: argparse.Namespace) -> SyntheticSplits:
    if config.key.startswith("synthetic") or config.key == "synthetic-newsvendor":
        return make_splits(
            seed=seed,
            n_train=args.n_train,
            n_learn=args.n_learn,
            n_recal=args.n_recal,
            n_test=args.n_test,
            y_dim=args.y_dim,
        )
    if config.key == "energy-linear":
        return make_energy_splits(seed=seed, cache_dir=args.data_dir)
    raise ValueError(f"No data generator registered for {config.key}.")


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _scenario_cost(args: argparse.Namespace, y_dim: int) -> np.ndarray:
    base = np.asarray(args.cost, dtype=float)
    if len(base) >= y_dim:
        return base[:y_dim]
    out = np.zeros(y_dim)
    out[: len(base)] = base
    return out


def _scenario_args(args: argparse.Namespace, scenario_id: str, y_dim: int, k_poly: int, n_learn: int, n_recal: int) -> argparse.Namespace:
    scenario = argparse.Namespace(**vars(args))
    scenario.scenario = scenario_id
    scenario.y_dim = int(y_dim)
    scenario.k_poly = int(k_poly)
    scenario.n_learn = int(n_learn)
    scenario.n_recal = int(n_recal)
    return scenario


def _sweep_scenarios(args: argparse.Namespace) -> list[argparse.Namespace]:
    y_dims = _parse_int_list(args.y_dims)
    k_values = _parse_int_list(args.k_polys)
    n_values = _parse_int_list(args.n_learns)
    m_values = _parse_int_list(args.n_recals)
    seen = set()
    scenarios = []
    for y_dim, k_poly, n_learn, n_recal in product(y_dims, k_values, n_values, m_values):
        key = (y_dim, k_poly, n_learn, n_recal)
        if key in seen:
            continue
        seen.add(key)
        sid = f"d{y_dim}_K{k_poly}_n{n_learn}_m{n_recal}"
        scenarios.append(_scenario_args(args, sid, y_dim, k_poly, n_learn, n_recal))

    if args.allocation_budgets:
        budgets = _parse_int_list(args.allocation_budgets)
        fractions = _parse_float_list(args.allocation_n_fracs)
        for y_dim, k_poly, budget, frac in product(y_dims, k_values, budgets, fractions):
            n_learn = int(round(budget * frac))
            n_learn = min(max(n_learn, 1), budget - 1)
            n_recal = budget - n_learn
            key = (y_dim, k_poly, n_learn, n_recal)
            if key in seen:
                continue
            seen.add(key)
            sid = f"d{y_dim}_K{k_poly}_n{n_learn}_m{n_recal}"
            scenarios.append(_scenario_args(args, sid, y_dim, k_poly, n_learn, n_recal))
    return scenarios


def _load_existing_results(path: Path) -> set[tuple]:
    """Load (experiment, dataset, loss, quadratic_weight, scenario, seed, method) keys from CSV."""
    existing: set[tuple] = set()
    if not path.exists():
        return existing
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    key = (
                        row["experiment"],
                        row["dataset"],
                        row["loss"],
                        float(row["quadratic_weight"]),
                        row["scenario"],
                        int(row["seed"]),
                        row["method"],
                    )
                    existing.add(key)
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return existing


def _run_seed(
    config: ExperimentConfig,
    seed: int,
    args: argparse.Namespace,
    existing_keys: set[tuple] | None = None,
) -> tuple[list[dict], list]:
    """Run a single seed; return (rows, fitted_sets) for plotting."""
    splits = _make_experiment_splits(config, seed, args)
    f_hat = fit_ridge_predictor(splits.train.x, splits.train.y)

    r_learn = residuals(f_hat, splits.learn.x, splits.learn.y)
    r_recal = residuals(f_hat, splits.recal.x, splits.recal.y)
    r_test = residuals(f_hat, splits.test.x, splits.test.y)
    centers_learn = f_hat.predict(splits.learn.x)
    centers_recal = f_hat.predict(splits.recal.x)
    centers_test = f_hat.predict(splits.test.x)

    context = BaselineContext(
        alpha=args.alpha,
        y_dim=splits.train.y.shape[1],
        cost=_scenario_cost(args, splits.train.y.shape[1]),
        robust_center=centers_learn.mean(axis=0),
        selection_centers=centers_learn[: args.max_selection_centers],
        k_poly=args.k_poly,
        n_croms=args.n_croms,
        seed=seed,
        loss_spec=config.loss_spec,
        centers_learn=centers_learn,
        centers_recal=centers_recal,
        miqcp_time_limit=float(args.miqcp_time_limit),
    )

    rows = []
    fitted_sets = []
    methods = baseline_registry(include_unfinished=args.include_unfinished)
    methods = [
        method for method in methods
        if method.supported_experiments is None or config.name in method.supported_experiments
    ]
    if args.methods != "all":
        requested = {item.strip() for item in args.methods.split(",") if item.strip()}
        methods = [method for method in methods if method.name in requested]
    for method in methods:
        row_id = {
            "experiment": config.name,
            "dataset": config.dataset,
            "loss": config.loss_spec.name,
            "quadratic_weight": config.loss_spec.quadratic_weight,
            "scenario": args.scenario,
            "y_dim": args.y_dim,
            "k_poly": args.k_poly,
            "n_train": args.n_train,
            "n_learn": args.n_learn,
            "n_recal": args.n_recal,
            "n_test": args.n_test,
            "seed": seed,
            "method": method.name,
        }

        existing_key = (
            config.name,
            config.dataset,
            config.loss_spec.name,
            float(config.loss_spec.quadratic_weight),
            args.scenario,
            seed,
            method.name,
        )
        if existing_keys is not None and existing_key in existing_keys:
            print(f"  [skip] {method.name} (seed={seed}, scenario={args.scenario}) — already in CSV")
            continue

        row = dict(row_id)
        try:
            t0 = time.perf_counter()
            fit_data = splits.learn.y if getattr(method, "calibration_space", "residual") == "y" else r_learn
            method.fit(fit_data, context)
            # Methods that don't use r_learn get the full r_learn ∪ r_recal
            # pool for calibration so the n_learn sweep is fair.
            if getattr(method, "calibration_space", "residual") == "y":
                cal_data = splits.recal.y if getattr(method, "uses_learning_split", True) else np.vstack([splits.learn.y, splits.recal.y])
            else:
                cal_data = r_recal if getattr(method, "uses_learning_split", True) else np.vstack([r_learn, r_recal])
            fitted_set = method.calibrate(cal_data, context)
            elapsed = time.perf_counter() - t0
            # Ours and its no-recal ablation share one geometry fit (cached);
            # charge the full fit time to whichever row reused it.
            elapsed += float(fitted_set.metadata.get("fit_time_cached", 0.0) or 0.0)
            fitted_sets.append(fitted_set)
            robust_centers = np.zeros_like(centers_test) if fitted_set.metadata.get("space") == "y" else centers_test
            coverage_data = splits.test.y if fitted_set.metadata.get("space") == "y" else r_test
            robust_metrics = average_robust_metrics(
                fitted_set,
                centers=robust_centers,
                y_true=splits.test.y,
                cost=_scenario_cost(args, splits.train.y.shape[1]),
                loss_spec=config.loss_spec,
                max_points=args.max_robust_points,
            )
            volume = set_area(fitted_set)
            row.update(
                {
                    "coverage": empirical_coverage(
                        fitted_set,
                        coverage_data,
                        centers=centers_test if getattr(fitted_set, "coverage_fn", None) else None,
                    ),
                    "volume": volume,
                    **robust_metrics,
                    "time": elapsed,
                    "metadata": fitted_set.metadata,
                }
            )
        except (NotImplementedError, ValueError, RuntimeError, ImportError) as exc:
            print(f"  [error] {method.name} (seed={seed}): {exc}")
            row.update({key: np.nan for key in METRIC_KEYS})
            row.update({"n_robust_eval": 0.0, "metadata": {"error": str(exc)}})
        rows.append(row)
    return rows, fitted_sets


def _test_residuals(config: ExperimentConfig, seed: int, args: argparse.Namespace) -> np.ndarray:
    splits = _make_experiment_splits(config, seed, args)
    f_hat = fit_ridge_predictor(splits.train.x, splits.train.y)
    return residuals(f_hat, splits.test.x, splits.test.y)


def _aggregate_rows(per_seed_rows: list[dict], method_order: list[str]) -> list[dict]:
    accum: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    row_meta: dict[tuple, dict] = {}
    ordered_keys = []

    for row in per_seed_rows:
        key = (
            row["experiment"],
            row["dataset"],
            row["loss"],
            row["quadratic_weight"],
            row["scenario"],
            row["y_dim"],
            row["k_poly"],
            row["n_train"],
            row["n_learn"],
            row["n_recal"],
            row["n_test"],
            row["method"],
        )
        if key not in row_meta:
            row_meta[key] = {
                k: row[k]
                for k in ["experiment", "dataset", "loss", "quadratic_weight", *SCENARIO_KEYS, "method"]
            }
            ordered_keys.append(key)
        for metric in METRIC_KEYS:
            value = row.get(metric, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = float("nan")
            accum[key][metric].append(value)

    def sort_key(key: tuple) -> tuple[int, int, int]:
        experiment = key[0]
        scenario = key[4]
        method = key[-1]
        experiment_order = [k[0] for k in ordered_keys]
        scenario_order = [k[4] for k in ordered_keys]
        try:
            method_rank = method_order.index(method)
        except ValueError:
            method_rank = len(method_order) + _method_order_key(method)
        return experiment_order.index(experiment), scenario_order.index(scenario), method_rank

    agg_rows = []
    for key in sorted(row_meta, key=sort_key):
        agg = dict(row_meta[key])
        for metric in METRIC_KEYS:
            vals = [v for v in accum[key][metric] if not math.isnan(v)]
            agg[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(agg)
    return agg_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_experiment(
    config: ExperimentConfig,
    seed: int,
    fitted_sets: list,
    agg_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    residual_test = _test_residuals(config, seed, args)

    # Methods shown in both the sets figure and the comparison figure.
    # Ordered to match the exp.tex table grouping (task-agnostic → Yeh → ours).
    key_names = [
        "PCP",
        "Conformal box",
        "F-CROMS (10 random lines)",
        "F-CROMS (20 random lines)",
        "Yeh et al. (PICNN E2E)",
        "Ours (CCG grid)",
        "Ours (CCG free)",
        "Ours (GD, structured)",
        "Ours (GD, free)",
        "Ours (MIQCP/pinball)",
    ]
    key_name_set = set(key_names)
    key_sets = [s for s in fitted_sets if s.name in key_name_set]
    # Preserve the ordering defined above
    key_sets.sort(key=lambda s: key_names.index(s.name) if s.name in key_names else 999)

    plot_residual_sets(residual_test, key_sets, RESULTS_DIR / f"{config.name}_{args.scenario}_sets.png")

    avg_objectives = {
        row["method"]: row["avg_robust_objective_mean"]
        for row in agg_rows
        if row["experiment"] == config.name
        and row["scenario"] == args.scenario
        and row["method"] in key_name_set
        and not math.isnan(row["avg_robust_objective_mean"])
    }
    if key_sets and avg_objectives:
        ordered_objectives = {s.name: avg_objectives[s.name] for s in key_sets if s.name in avg_objectives}
        plot_paper_figure(
            residual_test,
            key_sets,
            avg_objectives=ordered_objectives,
            output=RESULTS_DIR / f"{config.name}_{args.scenario}_comparison.png",
        )


def _plot_sweep_summaries(agg_rows: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("volume_mean", "volume"),
        ("avg_robust_objective_mean", "avg robust objective"),
        ("avg_realized_regret_mean", "avg realized regret"),
        ("time_mean", "runtime (s)"),
    ]
    params = [("y_dim", "d"), ("k_poly", "K"), ("n_learn", "n"), ("n_recal", "m")]
    for experiment in sorted({row["experiment"] for row in agg_rows}):
        exp_rows = [row for row in agg_rows if row["experiment"] == experiment]
        for param, label in params:
            values = sorted({int(row[param]) for row in exp_rows})
            if len(values) <= 1:
                continue
            fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.4), sharex=True)
            axes = np.atleast_1d(axes)
            methods = sorted({row["method"] for row in exp_rows})
            method_colors: dict[str, object] = {}
            cmap = plt.get_cmap("tab10")
            for i, m in enumerate(methods):
                method_colors[m] = cmap(i % 10)
            for ax, (metric, ylabel) in zip(axes, metrics):
                is_objective_panel = metric == "avg_robust_objective_mean"
                for method in methods:
                    xs, ys, yerrs = [], [], []
                    oracle_xs, oracle_ys = [], []
                    for value in values:
                        candidates = [
                            row for row in exp_rows
                            if row["method"] == method and int(row[param]) == value
                        ]
                        mean, err = _mean_and_error(candidates, metric, metric.replace("_mean", "_std"))
                        if mean is not None:
                            xs.append(value)
                            ys.append(mean)
                            yerrs.append(err)
                        if is_objective_panel:
                            rl_mean, _ = _mean_and_error(candidates, "avg_realized_loss_mean", "avg_realized_loss_std")
                            rg_mean, _ = _mean_and_error(candidates, "avg_realized_regret_mean", "avg_realized_regret_std")
                            if rl_mean is not None and rg_mean is not None:
                                oracle_xs.append(value)
                                oracle_ys.append(rl_mean - rg_mean)
                    color = method_colors[method]
                    if xs:
                        ax.errorbar(
                            xs,
                            ys,
                            yerr=yerrs,
                            marker="o",
                            lw=1.6,
                            ms=4,
                            capsize=2.5,
                            elinewidth=0.9,
                            color=color,
                            label=method,
                        )
                    if is_objective_panel and oracle_xs:
                        ax.plot(oracle_xs, oracle_ys, ls="--", lw=1.2, color=color)
                ax.set_xlabel(label)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.25)
            axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False)
            fig.suptitle(f"{experiment}: sweep over {label}", y=1.03)
            fig.tight_layout()
            out = output_dir / f"{experiment}_sweep_{param}.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved sweep summary to {out}")


def _finite_or_nan(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _mean_and_error(rows: list[dict], mean_key: str, std_key: str | None = None) -> tuple[float | None, float]:
    means = [_finite_or_nan(row.get(mean_key)) for row in rows]
    means = [value for value in means if value is not None]
    if not means:
        return None, 0.0

    mean = float(np.mean(means))
    variances = []
    for row in rows:
        row_mean = _finite_or_nan(row.get(mean_key))
        if row_mean is None:
            continue
        row_std = _finite_or_nan(row.get(std_key)) if std_key else None
        variances.append((row_std or 0.0) ** 2 + (row_mean - mean) ** 2)
    return mean, float(math.sqrt(np.mean(variances))) if variances else 0.0


def _method_order_key(method: str) -> int:
    order = [
        "Bonferroni box",
        "Conformal box",
        "Conformal ball",
        "PCP",
        "F-CROMS (10 random lines)",
        "F-CROMS (20 random lines)",
        "Yeh et al. (PICNN E2E)",
        "Decision-agnostic polytope",
        "Size-aware conformal polytope",
        "Decision-aware polytope (no recal)",
        "Ours (CCG grid)",
        "Ours (Gurobi MIQCP)",
        "Ours (CCG free)",
        "Ours (GD, structured)",
        "Ours (GD, free)",
        "Ours (MIQCP/pinball)",
    ]
    try:
        return order.index(method)
    except ValueError:
        return len(order)


def _method_color_map(methods: list[str]):
    explicit = {
        # Task-agnostic baselines — greyscale family
        "Bonferroni box":    "#B8B8B8",
        "Conformal box":     "#7F7F7F",
        "Conformal ball":    "#3A3A3A",
        # PCP family — violet shades
        "PCP":               "#8B5CF6",
        "PCP (generative)":  "#5B21B6",   # darker violet for original
        # F-CROMS — blue family
        "F-CROMS (10 random lines)": "#6BAED6",
        "F-CROMS (20 random lines)": "#2171B5",
        # Yeh — amber/gold
        "Yeh et al. (PICNN E2E)": "#E6AB02",
        "Yeh et al. (conditional)":  "#B45309",   # darker amber for original
        # Ablation polytopes — three distinct non-red hues
        "Decision-agnostic polytope":        "#8FD6C5",   # lighter green-teal
        "Size-aware conformal polytope":     "#3CB371",   # medium sea green
        "Decision-aware polytope (no recal)":"#F4A0A0",   # light red (sibling of ours)
        # Our main method — crimson, visually dominant
        "Ours (MIQCP/pinball)": "crimson",
        # MIQCP cross-check point — distinct gold hue, reads as a validation overlay
        "Ours (Gurobi MIQCP)": "#B8860B",
    }
    fallback = ["#6E6E6E", "#A0A0A0", "#BAB0AC", "#9D755D", "#4C4C4C"]
    fallback_i = 0
    colors = {}
    for method in methods:
        if method in explicit:
            colors[method] = explicit[method]
        elif _is_ours_family(method):
            colors[method] = "crimson"
        else:
            colors[method] = fallback[fallback_i % len(fallback)]
            fallback_i += 1
    return colors


def _is_ours_family(method: str) -> bool:
    return method.startswith("Ours") or method in {
        "Decision-agnostic polytope",
        "Decision-aware polytope (no recal)",
        "Size-aware conformal polytope",
    }


def _ours_alpha(method: str) -> float:
    alphas = {
        "Decision-agnostic polytope": 0.18,
        "Decision-aware polytope (no recal)": 0.42,
        "Size-aware conformal polytope": 0.68,
        "Ours (Gurobi MIQCP)": 0.82,
        "Ours (MIQCP/pinball)": 1.00,
    }
    return alphas.get(method, 0.85 if method.startswith("Ours") else 0.95)


def _paper_point_alpha(method: str) -> float:
    if method == "Ours (MIQCP/pinball)":
        return 1.0
    if method == "Ours (Gurobi MIQCP)":
        return 0.88
    if method == "Decision-aware polytope (no recal)":
        return 0.42
    return 0.30


def _paper_point_zorder(method: str) -> int:
    if method == "Ours (MIQCP/pinball)":
        return 8
    if method == "Ours (Gurobi MIQCP)":
        return 7
    if method.startswith("Ours"):
        return 6
    return 3


def _scaled_error(error: float, n_seeds: int | None) -> float:
    if n_seeds and n_seeds > 1:
        return error / math.sqrt(n_seeds)
    return error


def _aligned_coverage_xlim(
    xs: np.ndarray,
    target: float,
    min_width: float = 0.016,
    target_fraction: float = 0.36,
) -> tuple[float, float]:
    """Return an x-axis range that contains xs while keeping target aligned.

    The returned interval places the target coverage level at the same
    horizontal fraction of every subplot, while the width adapts to each
    subplot's observed coverage range.
    """
    finite_xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if finite_xs.size == 0:
        width = min_width
    else:
        left_need = max(0.0, float(target - finite_xs.min())) / max(target_fraction, 1e-6)
        right_need = max(0.0, float(finite_xs.max() - target)) / max(1.0 - target_fraction, 1e-6)
        width = max(1.18 * max(left_need, right_need), min_width)
    return target - target_fraction * width, target + (1.0 - target_fraction) * width


def _paper_tradeoff_methods(rows: list[dict]) -> list[str]:
    """Methods shown in paper tradeoff plots."""
    allowed = {
        "Bonferroni box",
        "Conformal box",
        "Conformal ball",
        "PCP",
        "F-CROMS (10 random lines)",
        "F-CROMS (20 random lines)",
        "Yeh et al. (PICNN E2E)",
        "Decision-agnostic polytope",
        "Decision-aware polytope (no recal)",
        "Size-aware conformal polytope",
        "Ours (MIQCP/pinball)",
        "Ours (Gurobi MIQCP)",
    }
    return [
        method for method in sorted({row["method"] for row in rows}, key=_method_order_key)
        if method in allowed
    ]


_PAPER_RC = {
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
}

_PAPER_MAIN_SCENARIO = "d2_K4_n300_m300"


def _legend_columns_for_two_rows(n_items: int) -> int:
    return max(1, int(math.ceil(n_items / 2.0)))


def _add_tradeoff_legend(fig, handles: list, labels: list) -> None:
    by_label = dict(zip(labels, handles))
    ours = [label for label in labels if _is_ours_family(label)]
    baselines = [label for label in labels if label not in ours]
    split = int(math.ceil(len(baselines) / 2.0))
    columns = [baselines[:split], baselines[split:], ours]
    pcp_labels = ["PCP", "PCP (generative)"]
    for pcp_label in pcp_labels:
        for column in columns:
            if pcp_label in column:
                column.remove(pcp_label)
        if pcp_label in by_label:
            columns[1].insert(0, pcp_label)  # first column: conformal sets; PCP heads the second
    fcroms10 = "F-CROMS (10 random lines)"
    if fcroms10 in columns[0]:
        columns[0].remove(fcroms10)
        columns[1].insert(0, fcroms10)
    anchors = [0.045, 0.36, 0.62]  # second column clears the longest first-column label

    for col_labels, x_anchor in zip(columns, anchors):
        if not col_labels:
            continue
        fig.legend(
            [by_label[label] for label in col_labels],
            [_display_method_name(label) for label in col_labels],
            loc="upper left",
            bbox_to_anchor=(x_anchor, 0.145),
            ncol=1,
            frameon=False,
            handletextpad=0.5,
            borderaxespad=0.0,
        )


def _plot_size_cost_tradeoff(agg_rows: list[dict], output_dir: Path, n_seeds: int | None = None) -> None:
    """Paper plot: set size versus average worst-case cost for each task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PAPER_RC)

    rows = [
        row for row in agg_rows
        if _finite_or_nan(row.get("volume_mean")) is not None
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return

    task_labels = {
        "synthetic_linear": "Synthetic (Linear loss)",
        "synthetic_quadratic_regularized": "Synthetic (Quadratic-regularized)",
        "synthetic_newsvendor": "Synthetic (Newsvendor)",
        "energy_efficiency_linear": "Energy Efficiency",
    }
    tasks = sorted({row["experiment"] for row in rows}, key=lambda item: list(task_labels).index(item) if item in task_labels else 999)
    methods = _paper_tradeoff_methods(rows)
    colors = _method_color_map(methods)

    ncols = 2
    nrows = int(math.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.3, 7.2), squeeze=False)
    handles = []
    labels = []

    for ax, task in zip(axes.ravel(), tasks):
        all_task_rows = [row for row in rows if row["experiment"] == task]
        from collections import Counter
        scenario_counts = Counter(row["scenario"] for row in all_task_rows)
        canonical_scenario = scenario_counts.most_common(1)[0][0]
        task_rows = [row for row in all_task_rows if row["scenario"] == canonical_scenario]
        plotted = []
        for method in methods:
            method_rows = [row for row in task_rows if row["method"] == method]
            if not method_rows:
                continue
            x_mean, x_err = _mean_and_error(method_rows, "volume_mean", "volume_std")
            y_mean, y_err = _mean_and_error(method_rows, "avg_robust_objective_mean", "avg_robust_objective_std")
            if x_mean is None or y_mean is None:
                continue
            is_ours = _is_ours_family(method)
            marker = "o"
            markersize = 8.2 if method == "Ours (MIQCP/pinball)" else 5.3
            point_alpha = _paper_point_alpha(method)
            x_err = _scaled_error(x_err, n_seeds)
            y_err = _scaled_error(y_err, n_seeds)
            point = ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_err,
                yerr=y_err,
                fmt=marker,
                ms=markersize,
                color=colors[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                ecolor=colors[method],
                elinewidth=0.75,
                capsize=2.0,
                alpha=point_alpha,
                label=method,
                zorder=_paper_point_zorder(method),
            )
            plotted.append((x_mean, y_mean))
            if method not in labels:
                handles.append(point)
                labels.append(method)
        ax.set_title(task_labels.get(task, task.replace("_", " ")))
        ax.set_xlabel("Set size")
        ax.set_ylabel("Avg. worst-case cost")
        if plotted:
            xs = np.asarray([item[0] for item in plotted], dtype=float)
            ys = np.asarray([item[1] for item in plotted], dtype=float)
            x_span = max(float(xs.max() - xs.min()), 1e-6)
            y_span = max(float(ys.max() - ys.min()), 1e-6)
            ax.set_xlim(float(xs.min() - 0.18 * x_span), float(xs.max() + 0.18 * x_span))
            ax.set_ylim(float(ys.min() - 0.20 * y_span), float(ys.max() + 0.20 * y_span))
        ax.grid(alpha=0.22)

    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")

    _add_tradeoff_legend(fig, handles, labels)
    fig.tight_layout(rect=(0.03, 0.17, 0.97, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "paper_size_cost_tradeoff.png"
    out_pdf = output_dir / "paper_size_cost_tradeoff.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved size/cost tradeoff plot to {out_png}")


def _plot_coverage_cost_tradeoff(agg_rows: list[dict], output_dir: Path, n_seeds: int | None = None, alpha: float = 0.10) -> None:
    """Paper plot: empirical coverage versus average worst-case cost for each task."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PAPER_RC)

    rows = [
        row for row in agg_rows
        if _finite_or_nan(row.get("coverage_mean")) is not None
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return

    task_labels = {
        "synthetic_linear": "Synthetic (Linear loss)",
        "synthetic_quadratic_regularized": "Synthetic (Quadratic-regularized)",
        "synthetic_newsvendor": "Synthetic (Newsvendor)",
        "energy_efficiency_linear": "Energy Efficiency",
    }
    tasks = sorted({row["experiment"] for row in rows}, key=lambda item: list(task_labels).index(item) if item in task_labels else 999)
    methods = _paper_tradeoff_methods(rows)
    colors = _method_color_map(methods)

    ncols = 2
    nrows = int(math.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.3, 7.2), squeeze=False)
    handles = []
    labels = []

    for ax, task in zip(axes.ravel(), tasks):
        all_task_rows = [row for row in rows if row["experiment"] == task]
        # Pin to a single canonical scenario (most frequent) so that scenario
        # variation across sweeps doesn't inflate the coverage error bars.
        from collections import Counter
        scenario_counts = Counter(row["scenario"] for row in all_task_rows)
        canonical_scenario = scenario_counts.most_common(1)[0][0]
        task_rows = [row for row in all_task_rows if row["scenario"] == canonical_scenario]
        plotted = []
        for method in methods:
            method_rows = [row for row in task_rows if row["method"] == method]
            if not method_rows:
                continue
            x_mean, x_err = _mean_and_error(method_rows, "coverage_mean", "coverage_std")
            y_mean, y_err = _mean_and_error(method_rows, "avg_robust_objective_mean", "avg_robust_objective_std")
            if x_mean is None or y_mean is None:
                continue
            is_ours = _is_ours_family(method)
            markersize = 8.2 if method == "Ours (MIQCP/pinball)" else 5.3
            point_alpha = _paper_point_alpha(method)
            x_err = _scaled_error(x_err, n_seeds)
            y_err = _scaled_error(y_err, n_seeds)
            point = ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_err,
                yerr=y_err,
                fmt="o",
                ms=markersize,
                color=colors[method],
                markeredgecolor="white",
                markeredgewidth=0.45,
                ecolor=colors[method],
                elinewidth=0.75,
                capsize=2.0,
                alpha=point_alpha,
                label=method,
                zorder=_paper_point_zorder(method),
            )
            plotted.append((x_mean, y_mean))
            if method not in labels:
                handles.append(point)
                labels.append(method)
        ax.axvline(1.0 - alpha, color="#888888", ls=":", lw=1.0)
        ax.set_title(task_labels.get(task, task.replace("_", " ")))
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Avg. worst-case cost")
        if plotted:
            xs = np.asarray([item[0] for item in plotted], dtype=float)
            ys = np.asarray([item[1] for item in plotted], dtype=float)
            ax.set_xlim(*_aligned_coverage_xlim(xs, 1.0 - alpha))
            y_span = max(float(ys.max() - ys.min()), 1e-6)
            ax.set_ylim(float(ys.min() - 0.20 * y_span), float(ys.max() + 0.20 * y_span))
        ax.grid(alpha=0.22)

    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")

    _add_tradeoff_legend(fig, handles, labels)
    fig.tight_layout(rect=(0.03, 0.17, 0.97, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "paper_coverage_cost_tradeoff.png"
    out_pdf = output_dir / "paper_coverage_cost_tradeoff.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage/cost tradeoff plot to {out_png}")


def _plot_coverage_cost_bubble(
    agg_rows: list[dict],
    output_dir: Path,
    n_seeds: int | None = None,
    alpha: float = 0.10,
) -> None:
    """Coverage (x) vs WC cost (y) bubble chart; bubble area encodes set volume.

    Visual language matches paper_coverage_cost_tradeoff: same colour map,
    same legend layout, same RC params.  Bubble size replaces error bars as the
    third data dimension so the chart stays uncluttered.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    plt.rcParams.update(_PAPER_RC)

    paper_task_order = [
        "synthetic_linear",
        "synthetic_quadratic_regularized",
        "synthetic_newsvendor",
        "energy_efficiency_linear",
    ]
    rows = [
        r for r in agg_rows
        if r.get("scenario") == _PAPER_MAIN_SCENARIO
        and r.get("experiment") in paper_task_order
        if _finite_or_nan(r.get("coverage_mean")) is not None
        and _finite_or_nan(r.get("avg_robust_objective_mean")) is not None
        and _finite_or_nan(r.get("volume_mean")) is not None
    ]
    if not rows:
        return

    task_labels = {
        "synthetic_linear": "Synthetic (Linear loss)",
        "synthetic_quadratic_regularized": "Synthetic (Quadratic-regularized)",
        "synthetic_newsvendor": "Synthetic (Newsvendor)",
        "energy_efficiency_linear": "Energy Efficiency",
    }
    tasks = [task for task in paper_task_order if any(r["experiment"] == task for r in rows)]
    methods = _paper_tradeoff_methods(rows)
    colors = _method_color_map(methods)

    # Fixed paper scenario; global volume range maps to marker area in pt^2.
    # Superlinear scaling keeps truly compact sets visibly compact while still
    # making large-volume baselines prominent. The upper end of the range is
    # robust (90th percentile) rather than the raw max, so an occasional
    # poorly-shaped random-library draw (e.g. F-CROMS selecting a wide
    # candidate for one seed) does not compress every other bubble's size;
    # such outliers simply saturate at S_MAX instead of dominating the scale.
    all_vols = [float(r["volume_mean"]) for r in rows if r.get("volume_mean")]
    v_min = min(all_vols)
    v_max = max(float(np.percentile(all_vols, 90)), v_min + 1e-8)
    S_MIN, S_MAX = 32.0, 780.0

    def _vol_to_s(v: float) -> float:
        t = (v - v_min) / max(v_max - v_min, 1e-8)
        t = max(0.0, min(1.0, t)) ** 1.25
        return S_MIN + (S_MAX - S_MIN) * t

    ncols = 2
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.3, 7.2), squeeze=False)
    handles: list = []
    labels: list = []
    all_xs: list = []

    for ax, task in zip(axes.ravel(), tasks):
        task_rows = [r for r in rows if r["experiment"] == task]

        plotted = []
        for method in methods:
            mrs = [r for r in task_rows if r["method"] == method]
            if not mrs:
                continue
            x = float(mrs[0]["coverage_mean"])
            y = float(mrs[0]["avg_robust_objective_mean"])
            v = float(mrs[0]["volume_mean"])

            point_alpha = _paper_point_alpha(method)
            s = _vol_to_s(v)

            # No error bars: means only (seed-level spread is in the tables).
            ax.scatter(
                x, y,
                s=s,
                color=colors[method],
                edgecolors="white",
                linewidths=0.45,
                alpha=point_alpha,
                zorder=_paper_point_zorder(method),
            )
            plotted.append((x, y))

            if method not in labels:
                proxy = Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=colors[method],
                    markeredgecolor="white", markeredgewidth=0.45,
                    markersize=(
                        7.8 if method == "Ours (MIQCP/pinball)"
                        else 7.0 if method == "Ours (Gurobi MIQCP)"
                        else 6.5
                    ),
                    alpha=point_alpha,
                )
                handles.append(proxy)
                labels.append(method)

        ax.axvline(1.0 - alpha, color="#888888", ls=":", lw=1.0)
        ax.set_title(task_labels.get(task, task.replace("_", " ")))
        ax.set_xlabel("Empirical coverage")
        ax.set_ylabel("Avg. worst-case cost")
        if plotted:
            xs = np.asarray([p[0] for p in plotted], dtype=float)
            ys = np.asarray([p[1] for p in plotted], dtype=float)
            all_xs.append(xs)
            y_span = max(float(ys.max() - ys.min()), 1e-6)
            ax.set_xlim(*_aligned_coverage_xlim(xs, 1.0 - alpha))
            ax.set_ylim(float(ys.min() - 0.20 * y_span), float(ys.max() + 0.20 * y_span))
        ax.grid(alpha=0.22)

    # One shared x-range and tick spacing for every task subplot, computed from
    # the coverage values of all panels, so the nominal-coverage line sits at the
    # same horizontal position in every panel.
    active_axes = list(axes.ravel()[:len(tasks)])
    if all_xs:
        shared_xlim = _aligned_coverage_xlim(np.concatenate(all_xs), 1.0 - alpha)
        for ax in active_axes:
            ax.set_xlim(*shared_xlim)
            ax.xaxis.set_major_locator(MultipleLocator(0.01))

    for ax in axes.ravel()[len(tasks):]:
        ax.axis("off")

    size_refs = [0.5, 1.0, 6.0]
    size_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=min(15.0, max(4.0, math.sqrt(0.42 * _vol_to_s(float(v))))),
            alpha=0.40,
        )
        for v in size_refs
    ]
    volume_ax = axes.ravel()[1] if len(tasks) > 1 else axes.ravel()[0]
    volume_ax.legend(
        size_handles,
        [f"{v:.1f}" for v in size_refs],
        title="Volume",
        loc="upper right",
        bbox_to_anchor=(1.025, 1.0),
        ncol=1,
        frameon=True,
        framealpha=0.62,
        fancybox=False,
        edgecolor="#CFCFCF",
        facecolor="white",
        fontsize=10,
        title_fontsize=10,
        labelspacing=0.25,
        handletextpad=0.65,
        borderpad=0.35,
        borderaxespad=0.35,
    )

    # Method colour legend (same 3-column layout as coverage_cost_tradeoff)
    _add_tradeoff_legend(fig, handles, labels)

    fig.tight_layout(rect=(0.03, 0.17, 0.97, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "paper_coverage_cost_bubble.png"
    out_pdf = output_dir / "paper_coverage_cost_bubble.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved coverage/cost bubble plot to {out_png}")


def _plot_gap_verification(agg_rows: list[dict], output_dir: Path) -> None:
    """Plot empirical proxy gaps for Theorem~thm:gap.

    The population oracle value is unknown in experiments.  We therefore use a
    transparent proxy: for each (experiment, method, d, K), subtract the best
    observed aggregate robust objective over the available (n,m) sweep.  This
    gives a nonnegative excess robust objective that can be compared with the
    theorem's predicted dependence on n and m.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in agg_rows:
        val = _finite_or_nan(row.get("avg_robust_objective_mean"))
        if val is None:
            continue
        key = (row["experiment"], row["method"], int(row["y_dim"]), int(row["k_poly"]))
        groups[key].append(row)

    gap_rows = []
    for key, rows in groups.items():
        best = min(float(row["avg_robust_objective_mean"]) for row in rows)
        for row in rows:
            gap_row = dict(row)
            gap_row["gap_proxy"] = max(0.0, float(row["avg_robust_objective_mean"]) - best)
            gap_rows.append(gap_row)

    for experiment in sorted({row["experiment"] for row in gap_rows}):
        exp_rows = [row for row in gap_rows if row["experiment"] == experiment]
        for sweep_param, label, ref_power in [
            ("n_learn", "learning size n", -0.5),
            ("n_recal", "recalibration size m", -0.5),
        ]:
            values = sorted({int(row[sweep_param]) for row in exp_rows})
            if len(values) <= 1:
                continue
            fig, ax = plt.subplots(figsize=(6.3, 4.2))
            for method in sorted({row["method"] for row in exp_rows}):
                xs, ys, yerrs = [], [], []
                for value in values:
                    vals = [
                        float(row["gap_proxy"])
                        for row in exp_rows
                        if row["method"] == method and int(row[sweep_param]) == value
                    ]
                    if vals:
                        xs.append(value)
                        ys.append(float(np.mean(vals)))
                        yerrs.append(float(np.std(vals, ddof=0)))
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerrs,
                        marker="o",
                        lw=1.7,
                        ms=4,
                        capsize=2.5,
                        elinewidth=0.9,
                        label=method,
                    )
            positive = [line.get_ydata() for line in ax.lines]
            flat_positive = [float(y) for ys in positive for y in ys if float(y) > 0]
            if flat_positive:
                x_ref = np.asarray(values, dtype=float)
                scale = max(flat_positive) / (x_ref[0] ** ref_power)
                ax.plot(x_ref, scale * x_ref**ref_power, color="#555555", ls="--", lw=1.2, label=r"reference $x^{-1/2}$")
                ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("proxy suboptimality gap")
            ax.set_title(f"{experiment}: empirical proxy gap vs {label}")
            ax.grid(alpha=0.25, which="both")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=False)
            fig.tight_layout()
            out = output_dir / f"{experiment}_gap_proxy_vs_{sweep_param}.png"
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved gap verification plot to {out}")


def _plot_cuts_sensitivity(agg_rows: list[dict], output_dir: Path, alpha: float) -> None:
    """Two-panel sensitivity plot: coverage and WC loss vs number of cuts K.

    Only rows where k_poly varies are included.  Methods are split into
    'Ours' variants (main focus) and key baselines shown for reference.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Methods to highlight as "Ours"; rest treated as reference baselines.
    our_prefix = "Ours"
    reference_methods = [
        "Bonferroni box",
        "Conformal box",
        "Conformal ball",
        "F-CROMS (10 random lines)",
        "F-CROMS (20 random lines)",
    ]

    for experiment in sorted({row["experiment"] for row in agg_rows}):
        exp_rows = [row for row in agg_rows if row["experiment"] == experiment]
        k_values = sorted({int(row["k_poly"]) for row in exp_rows})
        if len(k_values) <= 1:
            continue

        all_methods = sorted({row["method"] for row in exp_rows})
        our_methods = [m for m in all_methods if m.startswith(our_prefix)]
        ref_methods = [m for m in all_methods if m in reference_methods]
        if not our_methods:
            continue

        cmap_ours = plt.get_cmap("tab10")
        cmap_ref = plt.get_cmap("Set2")
        colors: dict[str, object] = {}
        for i, m in enumerate(our_methods):
            colors[m] = cmap_ours(i % 10)
        for i, m in enumerate(ref_methods):
            colors[m] = cmap_ref(i % 8)

        fig, (ax_cov, ax_wc) = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=False)

        for ax, metric, ylabel in [
            (ax_cov, "coverage_mean", "empirical coverage"),
            (ax_wc, "avg_robust_objective_mean", "avg WC loss"),
        ]:
            for method in our_methods + ref_methods:
                is_ours = method.startswith(our_prefix)
                xs, ys, yerrs = [], [], []
                for k in k_values:
                    candidates = [
                        row for row in exp_rows
                        if row["method"] == method and int(row["k_poly"]) == k
                    ]
                    mean, err = _mean_and_error(candidates, metric, metric.replace("_mean", "_std"))
                    if mean is not None:
                        xs.append(k)
                        ys.append(mean)
                        yerrs.append(err)
                if not xs:
                    continue
                ax.errorbar(
                    xs, ys, yerr=yerrs,
                    marker="o" if is_ours else "s",
                    lw=1.8 if is_ours else 1.1,
                    ms=5 if is_ours else 3.5,
                    ls="-" if is_ours else "--",
                    alpha=1.0 if is_ours else 0.55,
                    capsize=2.5,
                    elinewidth=0.9,
                    color=colors[method],
                    label=method,
                )
            if ax is ax_cov:
                ax.axhline(1.0 - alpha, color="#333333", ls=":", lw=1.1,
                           label=f"target {1-alpha:.2f}")
            ax.set_xlabel("number of cuts $K$")
            ax.set_ylabel(ylabel)
            ax.set_xticks(k_values)
            ax.grid(alpha=0.25)

        handles, labels = [], []
        for ax in (ax_cov, ax_wc):
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                   ncol=min(5, len(handles)), fontsize=7, frameon=False)
        fig.suptitle(f"{experiment}: sensitivity to number of cuts $K$", y=1.03)
        fig.tight_layout()
        out = output_dir / f"{experiment}_cuts_sensitivity.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved cuts sensitivity plot to {out}")


def _plot_allocation_objective_coverage(agg_rows: list[dict], output_dir: Path, alpha: float) -> None:
    """Plot robust objective across learning/recalibration splits.

    For each fixed total calibration budget N=n+m, the x-axis is the learning
    allocation n.  Solid lines show the average robust objective; dashed lines
    of the same color show the oracle (hindsight) loss.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = "synthetic_linear"
    rows = [
        row for row in agg_rows
        if row.get("experiment") == experiment
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return

    geometry_groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        geometry_groups[(int(row["y_dim"]), int(row["k_poly"]))].append(row)

    for (y_dim, k_poly), group_rows in sorted(geometry_groups.items()):
        budgets = sorted({int(row["n_learn"]) + int(row["n_recal"]) for row in group_rows})
        budgets = [
            budget for budget in budgets
            if len({int(row["n_learn"]) for row in group_rows if int(row["n_learn"]) + int(row["n_recal"]) == budget}) > 1
        ]
        if not budgets:
            continue

        methods = sorted({row["method"] for row in group_rows})
        cmap = plt.get_cmap("tab10")
        colors = {method: cmap(i % 10) for i, method in enumerate(methods)}
        ncols = min(3, len(budgets))
        nrows = int(math.ceil(len(budgets) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False)
        objective_handles = []

        for ax, budget in zip(axes.ravel(), budgets):
            budget_rows = [row for row in group_rows if int(row["n_learn"]) + int(row["n_recal"]) == budget]
            for method in methods:
                xs, obj, obj_err = [], [], []
                oracle_xs, oracle_ys = [], []
                for n_value in sorted({int(row["n_learn"]) for row in budget_rows if row["method"] == method}):
                    n_rows = [
                        row for row in budget_rows
                        if row["method"] == method and int(row["n_learn"]) == n_value
                    ]
                    if not n_rows:
                        continue
                    obj_mean, obj_std = _mean_and_error(
                        n_rows,
                        "avg_robust_objective_mean",
                        "avg_robust_objective_std",
                    )
                    if obj_mean is None:
                        continue
                    xs.append(n_value)
                    obj.append(obj_mean)
                    obj_err.append(obj_std)
                    rl_mean, _ = _mean_and_error(n_rows, "avg_realized_loss_mean", "avg_realized_loss_std")
                    rg_mean, _ = _mean_and_error(n_rows, "avg_realized_regret_mean", "avg_realized_regret_std")
                    if rl_mean is not None and rg_mean is not None:
                        oracle_xs.append(n_value)
                        oracle_ys.append(rl_mean - rg_mean)
                if not xs:
                    continue
                line_obj = ax.errorbar(
                    xs,
                    obj,
                    yerr=obj_err,
                    marker="o",
                    lw=1.8,
                    ms=4,
                    capsize=2.5,
                    elinewidth=0.9,
                    color=colors[method],
                    label=method,
                )
                if oracle_xs:
                    ax.plot(oracle_xs, oracle_ys, ls="--", lw=1.2, color=colors[method])
                if budget == budgets[0]:
                    objective_handles.append(line_obj)
            ax.set_title(f"$n+m={budget}$")
            ax.set_xlabel("learning size n")
            ax.set_ylabel("avg robust objective")
            ax.grid(alpha=0.25)

        for ax in axes.ravel()[len(budgets):]:
            ax.axis("off")

        if objective_handles:
            fig.legend(
                handles=objective_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(4, max(1, len(objective_handles))),
                fontsize=7,
                frameon=False,
            )
        fig.suptitle(f"synthetic_linear: robust objective vs allocation, d={y_dim}, K={k_poly}", y=1.02)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        suffix = "" if len(geometry_groups) == 1 else f"_d{y_dim}_K{k_poly}"
        out = output_dir / f"synthetic_linear{suffix}_sweep_allocation_objective_coverage.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved allocation objective plot to {out}")


def _load_per_seed_csv(path: Path) -> list[dict]:
    """Load all rows from a per-seed CSV into a list of dicts."""
    rows = []
    if not path.exists():
        return rows
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
    return rows


def _latex_escape(text: object) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _display_method_name(method: str) -> str:
    replacements = {
        "Yeh et al. (PICNN E2E)":         "Yeh et al.",
        "Conformal box":                     r"Conformal ($\ell_\infty$)",
        "Conformal ball":                    r"Conformal ($\ell_2$)",
        "F-CROMS (10 random lines)":          "F-CROMS (10)",
        "F-CROMS (20 random lines)":          "F-CROMS (20)",
        "Decision-agnostic polytope":         "Polyhedron",
        "Size-aware conformal polytope":      "Polyhedron (min size)",
        "Decision-aware polytope (no recal)": "Ours (w/o recal)",
        "Ours (MIQCP/pinball)":                  "Ours",
        "Ours (Gurobi MIQCP)":                "Ours (MIQCP)",
    }
    return replacements.get(method, method)


def _tex_num(value: object, digits: int = 3, scale: float = 1.0) -> str:
    val = _finite_or_nan(value)
    if val is None:
        return "--"
    return f"{val * scale:.{digits}f}"


def _tex_mean_std(row: dict, metric: str, digits: int = 3, scale: float = 1.0) -> str:
    mean = _finite_or_nan(row.get(f"{metric}_mean"))
    std = _finite_or_nan(row.get(f"{metric}_std"))
    if mean is None:
        return "--"
    if std is None or std == 0.0:
        return f"${_tex_num(mean, digits, scale)}$"
    return f"${_tex_num(mean, digits, scale)}\\pm{_tex_num(std, digits, scale)}$"


_COMPACT_METHOD_NAMES = {
    "Bonferroni box":                    "Bonferroni box",
    "Conformal box":                     r"Conformal ($\ell_\infty$)",
    "Conformal ball":                    r"Conformal ($\ell_2$)",
    "PCP":                                "PCP",
    "F-CROMS (10 random lines)":         "CROMS-style (10)",
    "F-CROMS (20 random lines)":         "CROMS-style (20)",
    "Yeh et al. (PICNN E2E)":         "Yeh et al.",
    "Decision-agnostic polytope":        "Polyhedron",
    "Size-aware conformal polytope":     "Polyhedron (min size)",
    "Decision-aware polytope (no recal)":r"Ours (w/o recal)",
    "Ours (MIQCP/pinball)":                 r"\textbf{Ours}",
}


_MAIN_TABLE_GROUPS = [
    ["Bonferroni box", "Conformal box", "Conformal ball", "PCP"],
    [
        "F-CROMS (10 random lines)", "F-CROMS (20 random lines)",
        "Yeh et al. (PICNN E2E)",
        "Decision-agnostic polytope", "Size-aware conformal polytope",
    ],
    ["Decision-aware polytope (no recal)", "Ours (MIQCP/pinball)"],
]


def _tex_time(row: dict) -> str:
    mean = _finite_or_nan(row.get("time_mean"))
    if mean is None:
        return "--"
    if mean < 0.01:
        return r"$<0.01$"
    if mean < 1.0:
        return f"${mean:.2f}$"
    return f"${mean:.1f}$"


def _write_main_tables_tex(agg_rows: list[dict], output_dir: Path) -> None:
    """Write the combined synthetic-task table and the Energy Efficiency
    table in the layout of the paper, so the file can be \\input directly."""
    synthetic_task_labels = {
        "synthetic_linear":               "Linear objective",
        "synthetic_quadratic_regularized":"Quadratic-regularized objective",
        "synthetic_newsvendor":           "Newsvendor objective",
    }
    energy_task = "energy_efficiency_linear"
    paper_methods = [m for group in _MAIN_TABLE_GROUPS for m in group]

    rows = [
        row for row in agg_rows
        if row.get("experiment") in synthetic_task_labels or row.get("experiment") == energy_task
        if row.get("method") in paper_methods
        and row.get("scenario") == _PAPER_MAIN_SCENARIO
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return

    def _best_wc(task_rows: list[dict]) -> float:
        valid = {m for m in paper_methods if m != "Decision-aware polytope (no recal)"}
        return min(
            float(row["avg_robust_objective_mean"])
            for row in task_rows if row["method"] in valid
        )

    def _row_cells(row: dict, best_wc: float) -> list[str]:
        method = _COMPACT_METHOD_NAMES.get(row["method"], _latex_escape(row["method"]))
        wc = _tex_mean_std(row, "avg_robust_objective", digits=2, scale=10.0)
        wc_mean = _finite_or_nan(row.get("avg_robust_objective_mean"))
        if wc_mean is not None and abs(wc_mean - best_wc) <= 1e-9:
            wc = r"\textbf{" + wc + "}"
        reg = _tex_mean_std(row, "avg_realized_regret", digits=2, scale=10.0)
        return [
            method,
            _tex_mean_std(row, "coverage", digits=3),
            _tex_mean_std(row, "volume", digits=2),
            wc,
            reg,
            _tex_time(row),
        ]

    def _ordered_task_rows(task: str) -> list[dict]:
        by_method = {row["method"]: row for row in rows if row["experiment"] == task}
        return [by_method[m] for group in _MAIN_TABLE_GROUPS for m in group if m in by_method]

    parts = [
        "% Auto-generated by experiments/run_synthetic2d.py. Requires booktabs, multirow, and graphicx.",
        "",
    ]

    # ── Combined synthetic-task table (tab:combined_main_results) ─────────────
    synthetic_task_order = list(synthetic_task_labels)
    per_task_rows = {task: _ordered_task_rows(task) for task in synthetic_task_order}
    if any(per_task_rows.values()):
        parts.extend([
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Comparison across synthetic objectives. Mean\,$\pm$\,std over five seeds;"
            r" $n=m=300$, $K=4$, $1{-}\alpha=0.90$."
            r" WC cost and regret are in units of $\times 10^{-1}$;"
            r" time is total wall-clock seconds per run.}",
            r"\label{tab:combined_main_results}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"\textbf{Task}"
            r" & \textbf{Method} & \textbf{Coverage} & \textbf{Volume}"
            r" & \textbf{WC cost} & \textbf{Regret} & \textbf{Time (s)} \\",
            r"\midrule",
            "",
        ])
        for t_idx, task in enumerate(synthetic_task_order):
            task_rows = per_task_rows[task]
            if not task_rows:
                continue
            best_wc = _best_wc(task_rows)
            label = _latex_escape(synthetic_task_labels[task])
            parts.append(
                rf"\multirow{{{len(task_rows)}}}{{*}}{{\parbox{{2.3cm}}{{{label}}}}}"
            )
            groups_present = [g for g in _MAIN_TABLE_GROUPS if any(m in {r['method'] for r in task_rows} for m in g)]
            last_group_idx = None
            for row in task_rows:
                group_idx = next(i for i, g in enumerate(groups_present) if row["method"] in g)
                if last_group_idx is not None and group_idx != last_group_idx:
                    parts.append(r"\cmidrule(lr){2-7}")
                last_group_idx = group_idx
                parts.append("& " + " & ".join(_row_cells(row, best_wc)) + r" \\")
            if t_idx < len(synthetic_task_order) - 1:
                parts.append(r"\midrule")
            parts.append("")
        parts.extend([
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ])

    # ── Energy Efficiency table (tab:energy_efficiency_linear_auto) ───────────
    energy_rows = _ordered_task_rows(energy_task)
    if energy_rows:
        best_wc = _best_wc(energy_rows)
        parts.extend([
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\caption{Energy Efficiency (real data). Mean\,$\pm$\,std over five random partitions;"
            r" $n_{\rm train}=320$, $n_{\rm learn}=160$, $n_{\rm recal}=144$, $n_{\rm test}=144$,"
            r" $K=4$, and $1{-}\alpha=0.90$. No resampling or augmentation is used."
            r" WC cost and regret are in units of $\times10^{-1}$; time is total wall-clock seconds per run.}",
            r"\label{tab:energy_efficiency_linear_auto}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\textbf{Method} & \textbf{Coverage} & \textbf{Volume}"
            r" & \textbf{WC cost} ($\times10^{-1}$) & \textbf{Regret} ($\times10^{-1}$) & \textbf{Time (s)} \\",
            r"\midrule",
        ])
        groups_present = [g for g in _MAIN_TABLE_GROUPS if any(m in {r['method'] for r in energy_rows} for m in g)]
        last_group_idx = None
        for row in energy_rows:
            group_idx = next(i for i, g in enumerate(groups_present) if row["method"] in g)
            if last_group_idx is not None and group_idx != last_group_idx:
                parts.append(r"\midrule")
            last_group_idx = group_idx
            parts.append(" & ".join(_row_cells(row, best_wc)) + r" \\")
        parts.extend([
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ])

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "paper_main_tables.tex"
    out.write_text("\n".join(parts))
    print(f"Saved LaTeX main tables to {out}")


def _plot_hyperparam_n_sweep(agg_rows: list[dict], output_dir: Path) -> None:
    """Clean line plot of WC cost vs n (learning split size) for paper methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plt.rcParams.update(_PAPER_RC)

    paper_methods = list(_COMPACT_METHOD_NAMES.keys())
    rows = [
        r for r in agg_rows
        if r.get("experiment") == "synthetic_linear"
        and r.get("method") in paper_methods
        and _finite_or_nan(r.get("avg_robust_objective_mean")) is not None
        and _finite_or_nan(r.get("n_learn")) is not None
    ]
    if not rows:
        return

    methods_present = [m for m in paper_methods if any(r["method"] == m for r in rows)]
    colors = _method_color_map(methods_present)

    # Aggregate: for each (method, n_learn) keep mean WC cost across scenarios
    from collections import defaultdict
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        n = int(float(r["n_learn"]))
        data[r["method"]][n].append(float(r["avg_robust_objective_mean"]))

    ns_all = sorted({int(float(r["n_learn"])) for r in rows})

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for method in methods_present:
        if method not in data:
            continue
        ns = sorted(data[method])
        ys = [float(np.mean(data[method][n])) for n in ns]
        is_ours = method == "Ours (MIQCP/pinball)"
        lw = 2.2 if is_ours else 1.2
        ls = "-"
        zorder = 5 if is_ours else 3
        label = _COMPACT_METHOD_NAMES.get(method, method).replace(r"\textbf{", "").replace("}", "")
        ax.plot(ns, ys, color=colors[method], lw=lw, ls=ls,
                marker="o", ms=4.5 if is_ours else 3.0,
                zorder=zorder, label=label, alpha=0.92)

    ax.set_xlabel("Learning split size $n$")
    ax.set_ylabel("Avg. worst-case cost")
    ax.grid(alpha=0.22)
    ax.legend(
        loc="upper right", fontsize=9, frameon=False,
        ncol=2, handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "paper_hyperparam_n_sweep.png"
    out_pdf = output_dir / "paper_hyperparam_n_sweep.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved hyperparameter n sweep plot to {out_png}")


_ORACLE_METHOD = "Decision-aware polytope (no recal)"


def _plot_paper_hyperparameter_sweeps(agg_rows: list[dict], output_dir: Path, alpha: float = 0.10) -> None:
    """Paper appendix plot sweeping d, K, and n/(n+m).

    Shows Conformal box (grey, task-agnostic reference), Ours (crimson solid),
    and Population oracle (crimson dashed) = Decision-aware polyhedron (no recal),
    which uses all n+m data for shape fitting without a held-out recalibration split.
    This approximates θ* and verifies convergence of the suboptimality gap Δ → 0.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_PAPER_RC)

    _OURS = "Ours (MIQCP/pinball)"
    # method_order controls draw order; oracle drawn as dashed line, not errorbar
    method_order = [_ORACLE_METHOD, _OURS]

    rows = [
        row for row in agg_rows
        if row.get("experiment") == "synthetic_linear"
        and row.get("method") in method_order
        and _finite_or_nan(row.get("coverage_mean")) is not None
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return
    methods = [m for m in method_order if any(row["method"] == m for row in rows)]
    colors = _method_color_map(methods)

    def _select_one_per_x(candidates: list[dict], x_key: str) -> list[dict]:
        selected: dict[int, dict] = {}
        for row in candidates:
            selected[int(row[x_key])] = row
        return [selected[x] for x in sorted(selected)]

    def _d_rows(method: str) -> list[dict]:
        return _select_one_per_x([
            row for row in rows
            if row["method"] == method
            and int(row["n_learn"]) == 300
            and int(row["n_recal"]) == 300
            and int(row["k_poly"]) == 8
        ], "y_dim")

    # Ours with K scaled proportionally to d: d=2→K=8, d=3→K=12, d=4→K=16.
    _K_FOR_D = {2: 8, 3: 12, 4: 16}

    def _d_more_k_rows() -> list[dict]:
        result = []
        for d, k in sorted(_K_FOR_D.items()):
            matching = [
                row for row in rows
                if row["method"] == _OURS
                and int(row["y_dim"]) == d
                and int(row["n_learn"]) == 300
                and int(row["n_recal"]) == 300
                and int(row["k_poly"]) == k
            ]
            if matching:
                result.append(matching[0])
        return result

    def _k_rows(method: str) -> list[dict]:
        return _select_one_per_x([
            row for row in rows
            if row["method"] == method
            and int(row["n_learn"]) == 300
            and int(row["n_recal"]) == 300
            and int(row["y_dim"]) == 2
        ], "k_poly")

    def _allocation_rows(method: str) -> list[dict]:
        return sorted(
            [
                row for row in rows
                if row["method"] == method
                and int(row["y_dim"]) == 2
                and int(row["k_poly"]) == 4
                and int(row["n_learn"]) + int(row["n_recal"]) == 600
            ],
            key=lambda row: int(row["n_learn"]),
        )

    # Each sweep entry: (xlabel, param_key, rows_fn, x_fn, ours_legend_label, extra_overlays, row_title)
    # extra_overlays: list of (legend_label, color, linestyle, rows_fn_no_args)
    # The legend label for "Ours" is uniform across rows (it is the same method/color
    # throughout); which hyperparameters are held fixed in each row is stated once via
    # a per-row title instead of being encoded into a different legend string per row.
    _MORE_K_LABEL = r"Ours ($K{=}4d$)"
    sweeps = [
        ("$d$", "y_dim", _d_rows, lambda row: int(row["y_dim"]),
         "Ours",
         [(_MORE_K_LABEL, "crimson", "--", _d_more_k_rows)],
         r"Sweep outcome dimension $d$ (baseline $K=8$)"),
        ("$K$", "k_poly", _k_rows, lambda row: int(row["k_poly"]),
         "Ours", [],
         r"Sweep number of cuts $K$ ($d=2$)"),
        ("$n/(n+m)$", "allocation", _allocation_rows,
         lambda row: int(row["n_learn"]) / (int(row["n_learn"]) + int(row["n_recal"])),
         "Ours", [],
         r"Sweep learning fraction $n/(n+m)$ ($K=4$, $d=2$)"),
    ]
    metrics = [
        ("coverage", "Empirical Coverage",
         lambda row: float(row["coverage_mean"]),
         lambda row: float(row["coverage_std"])),
        ("volume", "Volume",
         lambda row: float(row["volume_mean"]),
         lambda row: float(row["volume_std"])),
        ("avg_robust_objective", "Avg. worst-case cost",
         lambda row: float(row["avg_robust_objective_mean"]),
         lambda row: float(row["avg_robust_objective_std"])),
    ]

    fig, axes = plt.subplots(len(sweeps), len(metrics), figsize=(10.2, 7.5), squeeze=False)
    legend_handles: list = []
    legend_labels: list[str] = []

    for i, (xlabel, _, rows_fn, x_fn, ours_legend_label, extra_overlays, row_title) in enumerate(sweeps):
        for j, (metric_key, ylabel, y_fn, yerr_fn) in enumerate(metrics):
            ax = axes[i, j]
            has_data = False
            all_xs: list[float] = []

            for method in methods:
                sweep_rows = rows_fn(method)
                if not sweep_rows:
                    continue
                has_data = True
                xs = np.asarray([x_fn(row) for row in sweep_rows], dtype=float)
                ys = np.asarray([y_fn(row) for row in sweep_rows], dtype=float)
                yerrs = np.asarray([yerr_fn(row) for row in sweep_rows], dtype=float)
                all_xs.extend(xs.tolist())

                is_ours = method == _OURS
                is_oracle = method == _ORACLE_METHOD
                color = colors[method]
                if is_oracle:
                    # Oracle: grey solid with round dot
                    oracle_eb = ax.errorbar(
                        xs, ys, yerr=yerrs,
                        ls="-", lw=1.6, color="#7F7F7F",
                        marker="o", ms=4.5, alpha=0.85, zorder=3,
                        elinewidth=0.65, capsize=2.0,
                    )
                    if "Ours (w/o recal)" not in legend_labels:
                        legend_handles.append(oracle_eb.lines[0])
                        legend_labels.append("Ours (w/o recal)")
                else:
                    lw = 2.0 if is_ours else 1.25
                    ms = 4.8 if is_ours else 3.6
                    alp = 1.0 if is_ours else 0.72
                    zo = 4 if is_ours else 2
                    line = ax.errorbar(
                        xs, ys, yerr=yerrs,
                        color=color, marker="o", lw=lw, ms=ms,
                        elinewidth=lw * 0.45, capsize=2.4, alpha=alp, zorder=zo,
                    )
                    disp = ours_legend_label if is_ours else _display_method_name(method)
                    if disp not in legend_labels:
                        legend_handles.append(line.lines[0])
                        legend_labels.append(disp)

            # Overlay extra lines specific to this sweep row (e.g. Ours with K∝d).
            for ov_label, ov_color, ov_ls, ov_rows_fn in extra_overlays:
                ov_rows = ov_rows_fn()
                if not ov_rows:
                    continue
                has_data = True
                xs = np.asarray([x_fn(row) for row in ov_rows], dtype=float)
                ys = np.asarray([y_fn(row) for row in ov_rows], dtype=float)
                yerrs = np.asarray([yerr_fn(row) for row in ov_rows], dtype=float)
                all_xs.extend(xs.tolist())
                ov_line = ax.errorbar(
                    xs, ys, yerr=yerrs,
                    color=ov_color, ls=ov_ls, lw=1.8, marker="o", ms=4.5,
                    elinewidth=0.8, capsize=2.4, alpha=0.9, zorder=5,
                )
                if ov_label not in legend_labels:
                    legend_handles.append(ov_line.lines[0])
                    legend_labels.append(ov_label)

            if not has_data:
                ax.axis("off")
                continue
            if metric_key == "coverage":
                ax.axhline(1.0 - alpha, color="#777777", ls=":", lw=1.0)
                ax.yaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: rf"{v * 100:.0f}\%")
                )
            elif metric_key == "volume":
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
            elif metric_key == "avg_robust_objective":
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if j == 0:
                ax.set_title(row_title, loc="left", fontsize=10.5, fontweight="bold", pad=8)
            if xlabel == "$n/(n+m)$":
                ticks = sorted(set(round(float(x), 4) for x in all_xs))
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{x:.2f}" for x in ticks])
            else:
                ticks = sorted(set(int(round(float(x))) for x in all_xs))
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(x) for x in ticks])
            ax.grid(alpha=0.22)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.005),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=15,
            handlelength=1.9,
            columnspacing=1.2,
        )

    fig.tight_layout(rect=(0.02, 0.065, 0.98, 1.0))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "paper_hyperparameter_sweeps.png"
    out_pdf = output_dir / "paper_hyperparameter_sweeps.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper hyperparameter sweep plot to {out_png}")


def _write_hyperparameter_tables_tex(agg_rows: list[dict], output_dir: Path) -> None:
    rows = [
        row for row in agg_rows
        if row.get("experiment") == "synthetic_linear"
        and row.get("method") == "Ours (MIQCP/pinball)"
        and _finite_or_nan(row.get("avg_robust_objective_mean")) is not None
    ]
    if not rows:
        return

    def _metric_cells(row: dict) -> list[str]:
        return [
            _tex_mean_std(row, "coverage", digits=3),
            _tex_mean_std(row, "volume", digits=2),
            _tex_mean_std(row, "avg_robust_objective", digits=3),
            _tex_time(row),
        ]

    def _summarize(title: str, label: str, selected_rows: list[dict], value_fn) -> list[str]:
        selected_rows = sorted(selected_rows, key=value_fn)
        if len(selected_rows) <= 1:
            return []
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            rf"\caption{{Synthetic linear hyperparameter sweep over {title} for our method. Entries are mean$\pm$std over seeds; time is total wall-clock seconds per run.}}",
            rf"\label{{tab:hyperparam_{label}_auto}}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            rf"\textbf{{{title}}} & \textbf{{Coverage}} & \textbf{{Set size}} & \textbf{{WC cost}} & \textbf{{Time (s)}} \\",
            r"\midrule",
        ]
        for row in selected_rows:
            value = value_fn(row)
            if isinstance(value, float):
                value_label = f"{value:.2f}"
            else:
                value_label = str(value)
            lines.append(value_label + " & " + " & ".join(_metric_cells(row)) + r" \\")
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
        ])
        return lines

    d_rows = [
        row for row in rows
        if int(row["n_learn"]) == 300 and int(row["n_recal"]) == 300 and int(row["k_poly"]) == 8
    ]
    k_rows = [
        row for row in rows
        if int(row["n_learn"]) == 300 and int(row["n_recal"]) == 300 and int(row["y_dim"]) == 2
    ]
    allocation_rows = [
        row for row in rows
        if int(row["y_dim"]) == 2
        and int(row["k_poly"]) == 4
        and int(row["n_learn"]) + int(row["n_recal"]) == 600
    ]

    parts = ["% Auto-generated hyperparameter tables.", ""]
    parts.extend(_summarize("outcome dimension $d$", "d", d_rows, lambda row: int(row["y_dim"])))
    parts.extend(_summarize("number of cuts $K$", "K", k_rows, lambda row: int(row["k_poly"])))
    parts.extend(_summarize(
        "learning fraction $n/(n+m)$",
        "allocation",
        allocation_rows,
        lambda row: int(row["n_learn"]) / (int(row["n_learn"]) + int(row["n_recal"])),
    ))
    if len(parts) <= 2:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "paper_hyperparameter_tables.tex"
    out.write_text("\n".join(parts))
    print(f"Saved LaTeX hyperparameter tables to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--experiments", type=str, default="synthetic-linear,synthetic-quad,energy-linear")
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method names, or 'all'.")
    parser.add_argument("--k-poly", type=int, default=4)
    parser.add_argument("--y-dim", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-learn", type=int, default=300)
    parser.add_argument("--n-recal", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--y-dims", type=str, default=None, help="Comma-separated sweep values for outcome dimension d.")
    parser.add_argument("--k-polys", type=str, default=None, help="Comma-separated sweep values for number of cuts K.")
    parser.add_argument("--n-learns", type=str, default=None, help="Comma-separated sweep values for learning calibration size n.")
    parser.add_argument("--n-recals", type=str, default=None, help="Comma-separated sweep values for recalibration size m.")
    parser.add_argument("--allocation-budgets", type=str, default=None, help="Comma-separated total calibration budgets N=n+m for dense allocation sweeps.")
    parser.add_argument("--allocation-n-fracs", type=str, default="0.15,0.25,0.35,0.50,0.65,0.75,0.85", help="Comma-separated learning fractions n/(n+m) used with --allocation-budgets.")
    parser.add_argument("--n-croms", type=int, default=10)
    parser.add_argument("--miqcp-time-limit", type=float, default=1800.0,
                        help="Gurobi wall-clock limit (s) per exact MIQCP solve in Ours; <=0 skips the MIQCP.")
    parser.add_argument("--max-selection-centers", type=int, default=120)
    parser.add_argument("--max-robust-points", type=int, default=300)
    parser.add_argument("--cost", nargs=2, type=float, default=(-0.08, 0.0))
    parser.add_argument("--quadratic-weight", type=float, default=0.08)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "main_agg.csv")
    parser.add_argument("--seed-output", type=Path, default=RESULTS_DIR / "main_by_seed.csv",
                        help="Per-seed CSV for main table experiments.")
    parser.add_argument("--hyperparam-seed-output", type=Path, default=RESULTS_DIR / "sweep_by_seed.csv",
                        help="Per-seed CSV for sweep experiments (used only with --plot-only).")
    parser.add_argument("--no-plots", action="store_true")
    # Efficiency flags
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (experiment, seed, scenario, method) rows already present in --seed-output.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load --seed-output and regenerate plots/aggregates without running any experiments.",
    )
    args = parser.parse_args()
    args.y_dims = args.y_dims or str(args.y_dim)
    args.k_polys = args.k_polys or str(args.k_poly)
    args.n_learns = args.n_learns or str(args.n_learn)
    args.n_recals = args.n_recals or str(args.n_recal)
    args.scenario = f"d{args.y_dim}_K{args.k_poly}_n{args.n_learn}_m{args.n_recal}"

    configs = _experiment_configs(args)
    scenarios = _sweep_scenarios(args)
    seeds = list(range(args.seed, args.seed + args.n_seeds))

    method_order = [method.name for method in baseline_registry(include_unfinished=args.include_unfinished)]
    if args.methods != "all":
        requested_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
        method_order = [name for name in method_order if name in requested_methods]

    # Load previously saved per-seed rows to (a) skip existing and (b) merge results.
    prior_rows = _load_per_seed_csv(args.seed_output)
    existing_keys: set[tuple] | None = None
    if args.skip_existing or args.plot_only:
        existing_keys = _load_existing_results(args.seed_output)
        print(f"Loaded {len(existing_keys)} existing results from {args.seed_output}")

    if args.plot_only:
        print("--plot-only: skipping all experiment runs, re-aggregating from saved CSV.")
        per_seed_rows = prior_rows
    else:
        print(f"Running experiments {[config.key for config in configs]} with seeds {seeds}")
        print(f"Scenarios: {[scenario.scenario for scenario in scenarios]}")

        new_rows: list[dict] = []
        last_fitted_by_config = {}

        for scenario in scenarios:
            for config in configs:
                if config.key in {"energy-linear", "synthetic-newsvendor"} and scenario.y_dim != 2:
                    print(f"Skipping {config.name} for {scenario.scenario}: experiment requires y_dim=2.")
                    continue
                print(f"\n=== {config.name} ({config.dataset}, {config.loss_spec.name}, {scenario.scenario}) ===")
                for i, seed in enumerate(seeds):
                    print(f"--- Seed {seed} ({i + 1}/{len(seeds)}) ---")
                    rows, fitted_sets = _run_seed(config, seed, scenario, existing_keys=existing_keys)
                    new_rows.extend(rows)
                    if fitted_sets:
                        last_fitted_by_config[(config.key, scenario.scenario)] = (seed, fitted_sets, scenario)

        # Always merge: prior rows + new rows.  Deduplicate by taking the last occurrence
        # of each (experiment, dataset, loss, qw, scenario, seed, method) key so that
        # re-runs naturally override stale entries without silently discarding prior work.
        def _dedup_rows(rows: list[dict]) -> list[dict]:
            seen: dict[tuple, dict] = {}
            for row in rows:
                try:
                    key = (
                        row.get("experiment", ""),
                        row.get("dataset", ""),
                        row.get("loss", ""),
                        str(row.get("quadratic_weight", "")),
                        row.get("scenario", ""),
                        str(row.get("seed", "")),
                        row.get("method", ""),
                    )
                except Exception:
                    key = id(row)
                seen[key] = row
            return list(seen.values())

        per_seed_rows = _dedup_rows(prior_rows + new_rows)

        # Write the full merged per-seed CSV (safe: deduplication prevents double-counting).
        if new_rows:
            per_seed_fieldnames = ID_KEYS + METRIC_KEYS + ["metadata"]
            _write_csv(args.seed_output, per_seed_rows, per_seed_fieldnames)
            print(f"Wrote {len(per_seed_rows)} total rows to {args.seed_output} ({len(new_rows)} new)")

    agg_rows = _aggregate_rows(per_seed_rows, method_order=method_order)

    agg_fieldnames = [
        "experiment",
        "dataset",
        "loss",
        "quadratic_weight",
        *SCENARIO_KEYS,
        "method",
    ] + [f"{metric}_{stat}" for metric in METRIC_KEYS for stat in ("mean", "std")]
    _write_csv(args.output, agg_rows, agg_fieldnames)

    hyperparam_rows: list[dict] = []
    if args.plot_only and args.hyperparam_seed_output.exists() and args.hyperparam_seed_output != args.seed_output:
        hyperparam_per_seed = _load_per_seed_csv(args.hyperparam_seed_output)
        if hyperparam_per_seed:
            hyperparam_rows = _aggregate_rows(hyperparam_per_seed, method_order=method_order)

    if not args.no_plots:
        is_sweep = len(scenarios) > 1
        last_fitted_by_config = locals().get("last_fitted_by_config", {})
        if not is_sweep:
            for config in configs:
                for scenario in scenarios:
                    key = (config.key, scenario.scenario)
                    if key not in last_fitted_by_config:
                        continue
                    if scenario.y_dim != 2:
                        continue
                    seed, fitted_sets, scenario_args = last_fitted_by_config[key]
                    _plot_experiment(config, seed, fitted_sets, agg_rows, scenario_args)
        else:
            print("Skipping per-scenario comparison/set plots for sweep run; aggregate sweep plots will be generated.")
        _plot_sweep_summaries(agg_rows, RESULTS_DIR)
        _plot_gap_verification(agg_rows, RESULTS_DIR)
        _plot_cuts_sensitivity(agg_rows, RESULTS_DIR, args.alpha)
        _plot_allocation_objective_coverage(agg_rows, RESULTS_DIR, args.alpha)
        _plot_size_cost_tradeoff(agg_rows, RESULTS_DIR, args.n_seeds)
        _plot_coverage_cost_tradeoff(agg_rows, RESULTS_DIR, args.n_seeds, args.alpha)
        _plot_coverage_cost_bubble(agg_rows, RESULTS_DIR, args.n_seeds, args.alpha)
        if hyperparam_rows:
            print(f"Generating hyperparameter plots from {args.hyperparam_seed_output}")
            _plot_sweep_summaries(hyperparam_rows, RESULTS_DIR)
            _plot_gap_verification(hyperparam_rows, RESULTS_DIR)
            _plot_cuts_sensitivity(hyperparam_rows, RESULTS_DIR, args.alpha)
            _plot_allocation_objective_coverage(hyperparam_rows, RESULTS_DIR, args.alpha)
            _plot_paper_hyperparameter_sweeps(hyperparam_rows, RESULTS_DIR, args.alpha)

    _write_main_tables_tex(agg_rows, PAPER_DIR)
    _write_hyperparameter_tables_tex(hyperparam_rows or agg_rows, PAPER_DIR)

    w = 42
    print(
        f"\n{'experiment':32s} {'method':{w}s} {'coverage':>18s} {'volume':>18s} "
        f"{'robust':>18s} {'regret':>18s} {'time(s)':>14s}"
    )
    print("-" * (32 + w + 18 * 4 + 14 + 6))
    for row in agg_rows:
        def _fmt(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if math.isnan(mean):
                return f"{'n/a':>18s}"
            return f"{mean:.3f}+/-{std:.3f}".rjust(18)

        def _fmt_time(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if math.isnan(mean):
                return f"{'n/a':>14s}"
            return f"{mean:.2f}+/-{std:.2f}".rjust(14)

        print(
            f"{(row['experiment'] + '/' + row['scenario'])[:32]:32s} {row['method']:{w}s}"
            f"{_fmt('coverage')}{_fmt('volume')}"
            f"{_fmt('avg_robust_objective')}{_fmt('avg_realized_regret')}"
            f"{_fmt_time('time')}"
        )

    print(f"\nSaved aggregate records to {args.output}")
    print(f"Saved per-seed records to {args.seed_output}")


if __name__ == "__main__":
    main()
