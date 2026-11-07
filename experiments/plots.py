from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
(_REPO_ROOT / "results" / ".mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .sets import SetResult, polytope_vertices


def _plot_set_boundary(ax, s: SetResult, **kwargs) -> None:
    """Draw the boundary of a single SetResult on ax."""
    if s.kind == "ellipse" and s.threshold is not None and s.chol_factor is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
        xy = s.threshold * (unit_circle @ s.chol_factor.T)
    elif s.kind == "ball" and s.radius is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 300)
        xy = np.column_stack([s.radius * np.cos(theta), s.radius * np.sin(theta)])
    elif s.W is not None and s.h is not None:
        verts = polytope_vertices(s.W, s.h)
        xy = np.vstack([verts, verts[0]])
    else:
        return
    ax.plot(xy[:, 0], xy[:, 1], **kwargs)


def plot_residual_sets(residuals: np.ndarray, fitted_sets: list[SetResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(
        residuals[:, 0],
        residuals[:, 1],
        s=10,
        c="#9a9a9a",
        alpha=0.28,
        linewidths=0,
        label="test residuals",
    )

    for fitted_set in fitted_sets:
        _plot_set_boundary(ax, fitted_set, color=fitted_set.color, lw=2.0, label=fitted_set.name)

    ax.set_xlabel("residual dimension 1")
    ax.set_ylabel("residual dimension 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paper_figure(
    residuals: np.ndarray,
    key_sets: list[SetResult],
    avg_objectives: dict[str, float],
    output: Path,
) -> None:
    """Two-panel paper figure: residual sets (left) + avg WC objective bar chart (right).

    avg_objectives maps method name to average worst-case objective over test set.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9.0, 3.8), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    # Left: residual scatter + set boundaries
    ax_l.scatter(
        residuals[:, 0], residuals[:, 1],
        s=5, c="#c0c0c0", alpha=0.25, linewidths=0, zorder=0,
    )
    for s in key_sets:
        _plot_set_boundary(ax_l, s, color=s.color, lw=2.0, label=s.name, zorder=3)
    ax_l.set_xlabel(r"residual $r_1$", fontsize=9)
    ax_l.set_ylabel(r"residual $r_2$", fontsize=9)
    ax_l.set_aspect("equal", adjustable="datalim")
    ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")

    # Right: horizontal bar chart of avg WC objective
    color_map = {s.name: s.color for s in key_sets}
    names = list(avg_objectives.keys())
    vals = [avg_objectives[n] for n in names]
    colors = [color_map.get(n, "#888888") for n in names]
    y_pos = np.arange(len(names))
    bars = ax_r.barh(y_pos, vals, color=colors, height=0.55, zorder=3)
    for bar, val in zip(bars, vals):
        ax_r.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8,
        )
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(names, fontsize=8)
    ax_r.invert_yaxis()
    ax_r.set_xlabel("avg. worst-case loss", fontsize=9)
    ax_r.set_xlim(left=min(vals) * 0.90, right=max(vals) * 1.08)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    pdf_path = output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved paper figure to {pdf_path}")
