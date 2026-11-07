# Experiments for "Learning Polyhedral Conformal Sets for Robust Optimization".
#
#   make paper        run all experiments, then make the figures and tables
#   make run-main     main comparison, all methods and tasks (Figure 3, Tables 2-3)
#   make run-sweeps   sensitivity to d, K and the n/m split (Figure 4)
#   make table1       exact MIQCP vs. pinball-loss approximation (Table 1)
#   make plots        figures and LaTeX tables from the saved CSVs
#
# Variables can be set on the command line, e.g.
#   make run-main MAIN_SEEDS=2 ENERGY_SEEDS=2 MIQCP_TIME_LIMIT=60

PYTHON   := python
# Single-threaded BLAS and torch: the PICNN baseline of Yeh et al. is only
# reproducible bit-for-bit with a fixed thread count (Gurobi uses Threads=1).
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

RESULTS  := results
PAPER    := $(RESULTS)
SEED     := 2026
MAIN_SEEDS   := 20
ENERGY_SEEDS := 80
N_SEEDS  := 5
MIQCP_TIME_LIMIT := 1800
ALPHA    := 0.10

MAIN_CSV  := $(RESULTS)/main_by_seed.csv
SWEEP_CSV := $(RESULTS)/sweep_by_seed.csv
MAIN_AGG  := $(RESULTS)/main_agg.csv
SWEEP_AGG := $(RESULTS)/sweep_agg.csv
MAIN_METHODS  := Bonferroni box,Conformal box,Conformal ball,PCP,F-CROMS (10 random lines),F-CROMS (20 random lines),Yeh et al. (PICNN E2E),Decision-agnostic polytope,Decision-aware polytope (no recal),Size-aware conformal polytope,Ours (MIQCP/pinball)
SWEEP_METHODS := Conformal box,Decision-aware polytope (no recal),Ours (MIQCP/pinball)

MAIN_ARGS = --k-poly 4 --y-dim 2 --n-learn 300 --n-recal 300 --n-test 1000 \
	--methods '$(MAIN_METHODS)' --miqcp-time-limit $(MIQCP_TIME_LIMIT) \
	--seed-output $(MAIN_CSV) --output $(MAIN_AGG) --no-plots --skip-existing

.PHONY: run-main
run-main:
	$(PYTHON) -m experiments.run_synthetic2d --experiments synthetic-linear,synthetic-quad,synthetic-newsvendor \
		--seed $(SEED) --n-seeds $(MAIN_SEEDS) $(MAIN_ARGS)
	$(PYTHON) -m experiments.run_synthetic2d --experiments energy-linear \
		--seed $(SEED) --n-seeds $(ENERGY_SEEDS) $(MAIN_ARGS)

# The d=2, K=4, n=m=300 point of Figure 4 is taken from $(MAIN_CSV).
SWEEP_ARGS = --experiments synthetic-linear --seed $(SEED) --n-seeds $(N_SEEDS) \
	--methods '$(SWEEP_METHODS)' --miqcp-time-limit $(MIQCP_TIME_LIMIT) \
	--seed-output $(SWEEP_CSV) --output $(SWEEP_AGG) --no-plots --skip-existing

.PHONY: run-sweeps
run-sweeps:
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dims 2 --k-polys 6,8,10,12 --n-learns 300 --n-recals 300
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dims 3 --k-polys 8,12 --n-learns 300 --n-recals 300
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dims 4 --k-polys 8,16 --n-learns 300 --n-recals 300
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dim 2 --k-poly 4 --n-learn 120 --n-recal 480
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dim 2 --k-poly 4 --n-learn 210 --n-recal 390
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dim 2 --k-poly 4 --n-learn 390 --n-recal 210
	$(PYTHON) -m experiments.run_synthetic2d $(SWEEP_ARGS) --y-dim 2 --k-poly 4 --n-learn 480 --n-recal 120

.PHONY: table1
table1:
	$(PYTHON) -m experiments.run_exact_vs_pinball --study exact --time-limit $(MIQCP_TIME_LIMIT)

.PHONY: plots
plots:
	$(PYTHON) -m experiments.plot_paper --main-csv $(MAIN_CSV) --sweep-csv $(SWEEP_CSV) \
		--output-dir $(RESULTS) --paper-dir $(PAPER) --alpha $(ALPHA) --n-seeds $(N_SEEDS)

.PHONY: paper
paper: run-main run-sweeps table1 plots

.PHONY: clean
clean:
	rm -f $(RESULTS)/*.png $(RESULTS)/*.pdf $(RESULTS)/*.tex $(RESULTS)/*.csv
