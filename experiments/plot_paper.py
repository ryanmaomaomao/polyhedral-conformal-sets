"""Generate all paper plots and tables from saved experiment CSVs.

Reads two CSV files produced by run_synthetic2d.py:
  - main_by_seed.csv   : main-table experiments (all 4 tasks, canonical params)
  - sweep_by_seed.csv  : hyperparameter sweep experiments (K, n, m, allocation)

Writes to --paper-dir (core tex outcomes live with the paper):
  paper_main_tables.tex
  paper_hyperparameter_tables.tex

Writes to --output-dir (plots and raw-data-derived figures):
  paper_size_cost_tradeoff.{png,pdf}
  paper_coverage_cost_tradeoff.{png,pdf}
  paper_coverage_cost_bubble.{png,pdf}
  paper_hyperparameter_sweeps.{png,pdf}
  {experiment}_sweep_{param}.png          (diagnostic sweep plots)
  {experiment}_cuts_sensitivity.png       (K sensitivity)
  {experiment}_gap_proxy_vs_*.png         (suboptimality gap verification)
  synthetic_linear_sweep_allocation_*.png (allocation sweep)

Usage:
    python -m experiments.plot_paper
    python -m experiments.plot_paper --main-csv path/to/main.csv
    python -m experiments.plot_paper --main-csv a.csv --sweep-csv b.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.baselines import baseline_registry
from experiments.run_synthetic2d import (
    PAPER_DIR,
    RESULTS_DIR,
    _aggregate_rows,
    _load_per_seed_csv,
    _method_order_key,
    _plot_allocation_objective_coverage,
    _plot_coverage_cost_tradeoff,
    _plot_coverage_cost_bubble,
    _plot_cuts_sensitivity,
    _plot_gap_verification,
    _plot_hyperparam_n_sweep,
    _plot_paper_hyperparameter_sweeps,
    _plot_size_cost_tradeoff,
    _plot_sweep_summaries,
    _write_hyperparameter_tables_tex,
    _write_main_tables_tex,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all paper plots and LaTeX tables from saved experiment CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--main-csv", type=Path, default=RESULTS_DIR / "main_by_seed.csv",
        help="Per-seed CSV from main-table experiments (all 4 tasks).",
    )
    parser.add_argument(
        "--sweep-csv", type=Path, default=RESULTS_DIR / "sweep_by_seed.csv",
        help="Per-seed CSV from hyperparameter sweep experiments.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="Directory to write all plots and raw-data-derived figures.",
    )
    parser.add_argument(
        "--paper-dir", type=Path, default=PAPER_DIR,
        help="Directory to write the LaTeX table files (core tex outcomes live with the paper).",
    )
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="Miscoverage rate (target coverage = 1 - alpha).")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of seeds used to scale error bars.")
    parser.add_argument(
        "--paper-only", action="store_true",
        help="Skip diagnostic sweep/gap plots; write only the paper scatter plots and tables.",
    )
    args = parser.parse_args()

    method_order = [m.name for m in baseline_registry()]

    # ── Load and aggregate main-table results ──────────────────────────────────
    main_rows: list[dict] = []
    main_per_seed: list[dict] = []
    if args.main_csv.exists():
        main_per_seed = _load_per_seed_csv(args.main_csv)
        main_rows = _aggregate_rows(main_per_seed, method_order=method_order)
        print(f"Loaded {len(main_per_seed)} per-seed rows from {args.main_csv} → {len(main_rows)} agg rows")
    else:
        print(f"Warning: main CSV not found at {args.main_csv}; paper tables will be empty.")

    # ── Load and aggregate sweep results ──────────────────────────────────────
    sweep_rows: list[dict] = []
    sweep_per_seed: list[dict] = []
    if args.sweep_csv == args.main_csv:
        sweep_rows = main_rows
    elif args.sweep_csv.exists():
        sweep_per_seed = _load_per_seed_csv(args.sweep_csv)
        sweep_rows = _aggregate_rows(sweep_per_seed, method_order=method_order)
        print(f"Loaded {len(sweep_per_seed)} per-seed rows from {args.sweep_csv} → {len(sweep_rows)} agg rows")
    else:
        print(f"Note: sweep CSV not found at {args.sweep_csv}; sweep plots will be skipped.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = args.paper_dir
    paper_dir.mkdir(parents=True, exist_ok=True)

    # ── Paper scatter plots (from main-table results) ─────────────────────────
    print("\n── Paper scatter plots ──")
    _plot_size_cost_tradeoff(main_rows, output_dir, n_seeds=args.n_seeds)
    _plot_coverage_cost_tradeoff(main_rows, output_dir, n_seeds=args.n_seeds, alpha=args.alpha)
    _plot_coverage_cost_bubble(main_rows, output_dir, n_seeds=args.n_seeds, alpha=args.alpha)

    # ── Paper LaTeX tables ────────────────────────────────────────────────────
    print("\n── LaTeX tables ──")
    _write_main_tables_tex(main_rows, paper_dir)
    _write_hyperparameter_tables_tex(sweep_rows or main_rows, paper_dir)

    # ── Hyperparameter sweep plot (replaces table in paper) ───────────────────
    print("\n── Hyperparameter sweep plot ──")
    # The sweep figure also needs the main-table synthetic-linear point
    # (d=2, K=4, n=m=300), which is run once as part of the main table.
    # The main table has more seeds than the sweeps (20 vs 5), so this point is
    # averaged over the sweep seeds only, like every other point of the figure.
    sweep_seeds = {row["seed"] for row in sweep_per_seed}
    main_point = [
        row for row in main_per_seed
        if row["experiment"] == "synthetic_linear" and (not sweep_seeds or row["seed"] in sweep_seeds)
    ]
    print(f"Sweep figure: main-table point uses seeds {sorted({row['seed'] for row in main_point})}")
    sweep_plot_rows = sweep_rows + _aggregate_rows(main_point, method_order=method_order)
    _plot_hyperparam_n_sweep(sweep_plot_rows or main_rows, output_dir)
    _plot_paper_hyperparameter_sweeps(sweep_plot_rows or main_rows, output_dir, alpha=args.alpha)

    if not args.paper_only:
        # ── Diagnostic / sweep plots (from sweep results) ─────────────────────
        diag_rows = sweep_rows or main_rows
        print("\n── Diagnostic plots ──")
        _plot_sweep_summaries(diag_rows, output_dir)
        _plot_gap_verification(diag_rows, output_dir)
        _plot_cuts_sensitivity(diag_rows, output_dir, alpha=args.alpha)
        _plot_allocation_objective_coverage(diag_rows, output_dir, alpha=args.alpha)

    print(f"\nAll paper outputs written to {output_dir}/")


if __name__ == "__main__":
    main()
