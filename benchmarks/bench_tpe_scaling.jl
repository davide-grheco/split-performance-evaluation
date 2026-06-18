#!/usr/bin/env julia
#
# bench_tpe_scaling.jl — Benchmark tpe_search() at increasing budgets and
# fit a linear model to extrapolate to the production budget (default 50).
#
# Runs tpe_search() at budget ∈ [1, 3, 5, 10] for each model.
# Fits time = intercept + slope × budget via OLS to estimate:
#   - per-trial cost (slope)
#   - fixed overhead (intercept: loading, TPE machinery, final refit)
# Then predicts cost at the target budget.
#
# Usage:
#   julia --project=. benchmarks/bench_tpe_scaling.jl \
#       [n_samples=2000] [n_features=2000] [n_inner_folds=5] [target_budget=50]
#
# Outputs:
#   - Markdown table on stdout
#   - benchmarks/results/tpe_scaling.csv

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
target_budget = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50

const MODEL_NAMES = ["lightgbm", "random_forest", "knn", "svr", "elasticnet"]
const BUDGETS     = [1, 3, 5, 10]

@info "bench_tpe_scaling" n_samples n_features n_inner_folds target_budget

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
rng   = MersenneTwister(42)
X_raw = rand(rng, 0:1, n_samples, n_features)
X     = coerce(MLJ.table(X_raw), Count => Continuous)
y     = randn(rng, n_samples)

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
@info "Warming up (150×30)…"
let
    rng_w = MersenneTwister(0)
    X_w   = coerce(MLJ.table(rand(rng_w, 0:1, 150, 30)), Count => Continuous)
    y_w   = randn(rng_w, 150)
    for name in MODEL_NAMES
        tpe_search(name, X_w, y_w; budget=1, n_inner_folds=2, seed=0)
    end
end

# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------
function ols_fit(xs::Vector{Float64}, ys::Vector{Float64})
    n    = length(xs)
    xm   = mean(xs)
    ym   = mean(ys)
    sxy  = sum((xs .- xm) .* (ys .- ym))
    sxx  = sum((xs .- xm).^2)
    slope = sxy / sxx
    intercept = ym - slope * xm
    return intercept, slope
end

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
@info "Benchmarking…"

raw_rows = DataFrame(
    model    = String[],
    budget   = Int[],
    time_s   = Float64[],
    alloc_gb = Float64[],
)

for name in MODEL_NAMES
    @info "  model=$name"
    for b in BUDGETS
        GC.gc()
        _, t, bytes, _, _ = @timed tpe_search(name, X, y;
            budget=b, n_inner_folds=n_inner_folds, seed=42)
        push!(raw_rows, (name, b, t, bytes / 1e9))
        @info "    budget=$b" time_s=round(t; digits=3)
    end
end

# ---------------------------------------------------------------------------
# Fit OLS per model and extrapolate
# ---------------------------------------------------------------------------
summary_rows = DataFrame(
    model         = String[],
    n_samples     = Int[],
    n_features    = Int[],
    n_folds       = Int[],
    intercept_s   = Float64[],
    slope_s       = Float64[],
    est_target_s  = Float64[],
    target_budget = Int[],
)

for name in MODEL_NAMES
    sub  = filter(r -> r.model == name, raw_rows)
    xs   = Float64.(sub.budget)
    ys   = sub.time_s
    intercept, slope = ols_fit(xs, ys)
    est  = intercept + slope * target_budget
    push!(summary_rows, (name, n_samples, n_features, n_inner_folds,
                         intercept, slope, est, target_budget))
end

# ---------------------------------------------------------------------------
# Print raw measurements table
# ---------------------------------------------------------------------------
println()
println("## TPE Scaling — Raw Measurements")
println("n_samples=$n_samples  n_features=$n_features  n_folds=$n_inner_folds")
println()
@printf("%-16s %8s %10s %10s\n", "model", "budget", "time_s", "alloc_GB")
println(repeat("-", 50))
for r in eachrow(raw_rows)
    @printf("%-16s %8d %10.3f %10.3f\n", r.model, r.budget, r.time_s, r.alloc_gb)
end

# ---------------------------------------------------------------------------
# Print extrapolation table
# ---------------------------------------------------------------------------
println()
println("## TPE Scaling — OLS Extrapolation to budget=$target_budget")
println()
@printf("%-16s %12s %12s %16s %16s\n",
    "model", "intercept_s", "slope_s/trial", "est_s", "est_min")
println(repeat("-", 78))
for r in eachrow(summary_rows)
    @printf("%-16s %12.3f %12.3f %16.1f %16.2f\n",
        r.model, r.intercept_s, r.slope_s, r.est_target_s, r.est_target_s / 60)
end
println()

seq_total = sum(summary_rows.est_target_s)
par_total = maximum(summary_rows.est_target_s)
@printf("Sequential total (all models):       %8.1f s  (%5.1f min)\n", seq_total, seq_total / 60)
@printf("Ideal parallel wall time (Nthreads≥%d): %8.1f s  (%5.1f min)\n",
    nrow(summary_rows), par_total, par_total / 60)

# ---------------------------------------------------------------------------
# Write CSVs
# ---------------------------------------------------------------------------
out_dir = joinpath(@__DIR__, "results")
mkpath(out_dir)

CSV.write(joinpath(out_dir, "tpe_scaling_raw.csv"),     raw_rows)
CSV.write(joinpath(out_dir, "tpe_scaling_summary.csv"), summary_rows)
@info "Results written" dir=out_dir
