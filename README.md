# Train–Test Splits Matter More for Evaluation Than for Performance

Code and reproducibility guide for the Merck molecular activity splitting benchmark.

- **Manuscript**: submitted
- **Data & results**: [doi:10.5281/zenodo.20705244](https://doi.org/10.5281/zenodo.20705244) — pre-computed CV splits and all experiment outputs
- **Source datasets**: [doi:10.1021/ci500747n](https://doi.org/10.1021/ci500747n) (Merck Supporting Information)

---

## Quick start

### 1. Get the Merck datasets

Download the Supporting Information from https://doi.org/10.1021/ci500747n and place the CSVs in `Data/merck/`:

```
Data/merck/<DATASET>_training_disguised.csv
Data/merck/<DATASET>_test_disguised.csv
```

where `<DATASET>` is one of: `3A4 CB1 DPP4 HIVINT HIVPROT LOGD METAB NK1 OX1 OX2 PGP PPB RAT_F TDI THROMBIN`.

### 2. Install Julia dependencies

Requires Julia 1.12.x ([julialang.org/downloads](https://julialang.org/downloads/)).

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

### 3. Download pre-computed data from Zenodo

CV folds and experiment outputs are archived at [doi:10.5281/zenodo.20705244](https://doi.org/10.5281/zenodo.20705244).
Download and extract into the repository root:

```bash
# CV folds (needed to re-run experiments with the same splits)
curl -L https://zenodo.org/records/20705244/files/splits_merck.tar.gz | tar -xz

# Experiment results (needed to regenerate figures without re-running)
curl -L https://zenodo.org/records/20705244/files/results_revision_v1.tar.gz | tar -xz
curl -L https://zenodo.org/records/20705244/files/results_revision_v1_sub1000.tar.gz | tar -xz
curl -L https://zenodo.org/records/20705244/files/results_revision_v1_sub150.tar.gz | tar -xz
```

---

## Regenerate article figures

With experiment results present under `experiments/`:

```bash
# Primary figures (revision_v1)
EXPERIMENT=revision_v1 FIG_DIR=images/revision_v1 \
  julia --project=. scripts/generate_article_figures.jl

# Sub-1000 sensitivity
EXPERIMENT=revision_v1_sub1000 FIG_DIR=images/revision_v1_sub1000 \
  SKIP_CHAMPION=1 julia --project=. scripts/generate_article_figures.jl

# Sub-150 sensitivity
EXPERIMENT=revision_v1_sub150 FIG_DIR=images/revision_v1_sub150 \
  SKIP_CHAMPION=1 julia --project=. scripts/generate_article_figures.jl

# Multicriteria decision analysis (TOPSIS/MOORA overall ranking + trade-off figure)
julia --project=. scripts/generate_mcda.jl
```

---

## Re-run experiments from scratch

Requires Snakemake (`pip install snakemake`), the Merck CSVs in `Data/merck/`, and the CV folds extracted under `Splits/` (see step 3 above).

```bash
EXPERIMENT_CONFIG=config/experiment.toml \
  snakemake -s Snakefile.experiment --cores 8
```

Environment variables: `DATA_ROOT` (default `Data/merck`), `CV_ROOT`/`OUT_ROOT` (default `Splits/merck`), `FRAC=0.8`, `BASE_SEED=42`, `REPEATS=5`, `KFOLDS=5`.

---

## Repository layout

```
Data/merck/          — raw Merck CSVs (not in git; download from doi:10.1021/ci500747n)
Splits/merck/        — outer CV folds per dataset (download from Zenodo)
experiments/         — per-experiment outputs: metrics, predictions, AD (download from Zenodo)
  revision_v1/       — primary results (full data)
  revision_v1_sub1000/ — Sub-1000 sensitivity
  revision_v1_sub150/  — Sub-150 sensitivity
images/              — generated figures (PDF/TIFF) for the manuscript
src/                 — Julia library: splitters, metrics, statistical tests, plotting
scripts/             — Julia entrypoints: run_split.jl, generate_article_figures.jl, …
config/              — experiment TOML configs
```

## Running tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
