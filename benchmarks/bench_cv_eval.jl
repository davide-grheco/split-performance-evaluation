#!/usr/bin/env julia
#
# bench_cv_eval.jl — Benchmark the cost of one inner-CV evaluation per model.
#
# One TPE trial = one call to MLJ.evaluate(..., CV(nfolds=K)).
# We approximate this by running tpe_search(..., budget=1), which does
# exactly 1 trial + 1 final refit. The trial cost dominates at realistic K.
#
# Usage:
#   julia --project=. benchmarks/bench_cv_eval.jl \
#       [n_samples=2000] [n_features=2000] [n_inner_folds=5] [n_reps=3]
#
# Outputs:
#   - Markdown table on stdout with median time, memory, and extrapolated
#     estimates for a full budget=50 / n_folds=5 TPE run.
#   - benchmarks/results/cv_eval.csv

using DataSplitBench
using MLJ
using Random
using Statistics
using Printf
using CSV
using DataFrames

n_samples     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000
n_features    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2000
n_inner_folds = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
n_reps        = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
# Optional 5th arg: comma-separated model names to benchmark (e.g. "lightgbm,knn")
const _ALL_MODELS = ["lightgbm", "random_forest", "knn", "svr", "elasticnet"]
const MODEL_NAMES = length(ARGS) >= 5 ?
    filter(m -> m in _ALL_MODELS, String.(split(ARGS[5], ","))) :
    _ALL_MODELS

@info "bench_cv_eval" n_samples n_features n_inner_folds n_reps

# ---------------------------------------------------------------------------
# Data — binary integer features matching Merck molecular fingerprints
# ---------------------------------------------------------------------------
rng   = MersenneTwister(42)
X_raw = rand(rng, 0:1, n_samples, n_features)
X     = coerce(MLJ.table(X_raw), Count => Continuous)
y     = randn(rng, n_samples)

# ---------------------------------------------------------------------------
# Warmup — absorb JIT compilation before timing.
# Always uses a tiny fixed dataset so this stays fast regardless of input size.
# ---------------------------------------------------------------------------
@info "Warming up (budget=1, n_folds=2, 150×30)…"
let
    rng_w = MersenneTwister(0)
    X_w   = coerce(MLJ.table(rand(rng_w, 0:1, 150, 30)), Count => Continuous)
    y_w   = randn(rng_w, 150)
    for name in MODEL_NAMES
        tpe_search(name, X_w, y_w; budget=1, n_inner_folds=2, seed=0)
    end
end

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@info "Benchmarking…"

rows = DataFrame(
    model          = String[],
    n_samples      = Int[],
    n_features     = Int[],
    n_folds        = Int[],
    n_reps         = Int[],
    median_s       = Float64[],
    min_s          = Float64[],
    max_s          = Float64[],
    alloc_gb       = Float64[],
    est_budget50_s = Float64[],
)

for name in MODEL_NAMES
    times  = Float64[]
    allocs = Float64[]
    for rep in 1:n_reps
        GC.gc()
        _, t, bytes, _, _ = @timed tpe_search(name, X, y;
            budget=1, n_inner_folds=n_inner_folds, seed=rep)
        push!(times,  t)
        push!(allocs, bytes / 1e9)
    end

    med_t  = median(times)
    min_t  = minimum(times)
    max_t  = maximum(times)
    med_gb = median(allocs)

    # Extrapolation: budget=50 repeats the inner loop 50 times.
    # budget=1 ≈ 1 trial + 1 refit; the refit is included but minor.
    # Conservative estimate: 50 × median time of budget=1.
    est_50 = 50.0 * med_t

    push!(rows, (name, n_samples, n_features, n_inner_folds, n_reps,
                 med_t, min_t, max_t, med_gb, est_50))
    @info "  done" model=name median_s=round(med_t; digits=3) alloc_gb=round(med_gb; digits=3)
end

# ---------------------------------------------------------------------------
# Print Markdown table
# ---------------------------------------------------------------------------
println()
println("## CV Eval Benchmark")
println("n_samples=$n_samples  n_features=$n_features  n_folds=$n_inner_folds  n_reps=$n_reps")
println()
@printf("%-16s %10s %10s %10s %16s %16s\n",
    "model", "median_s", "min_s", "alloc_GB", "est_b50_s", "est_b50_min")
println(repeat("-", 84))
for r in eachrow(rows)
    @printf("%-16s %10.3f %10.3f %10.3f %16.1f %16.2f\n",
        r.model, r.median_s, r.min_s, r.alloc_gb,
        r.est_budget50_s, r.est_budget50_s / 60)
end
println()

seq_total = sum(rows.est_budget50_s)
par_total = maximum(rows.est_budget50_s)
@printf("Sequential total (all models):       %8.1f s  (%5.1f min)\n", seq_total, seq_total / 60)
@printf("Ideal parallel wall time (Nthreads≥%d): %8.1f s  (%5.1f min)\n",
    nrow(rows), par_total, par_total / 60)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
out_dir  = joinpath(@__DIR__, "results")
out_path = joinpath(out_dir, "cv_eval.csv")
mkpath(out_dir)
CSV.write(out_path, rows)
@info "Results written" path=out_path
