#!/usr/bin/env julia
#
# bench_data_scaling.jl — Benchmark tpe_search() across data sizes.
#
# Merck datasets span ~1.8K–37K training samples and 4K–9K features.
# This script measures how time and memory scale with n_samples at a
# fixed feature count, using a small budget to stay fast.
#
# Usage:
#   julia --project=. benchmarks/bench_data_scaling.jl \
#       [n_features=2000] [n_inner_folds=2] [budget=3]
#
# Outputs:
#   - Markdown table on stdout
#   - benchmarks/results/data_scaling.csv

using DataSplitBench
using MLJ
using Random
using Statistics
using Printf
using CSV
using DataFrames

n_features    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2000
n_inner_folds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
budget        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3

const MODEL_NAMES = ["lightgbm", "random_forest", "knn", "svr", "elasticnet"]
const SAMPLE_SIZES = [1_000, 5_000, 10_000, 20_000]

@info "bench_data_scaling" n_features n_inner_folds budget

# ---------------------------------------------------------------------------
# Warmup with smallest size
# ---------------------------------------------------------------------------
@info "Warming up…"
let
    rng   = MersenneTwister(0)
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
    model      = String[],
    n_samples  = Int[],
    n_features = Int[],
    n_folds    = Int[],
    budget     = Int[],
    time_s     = Float64[],
    alloc_gb   = Float64[],
)

for n_samples in SAMPLE_SIZES
    @info "  n_samples=$n_samples"
    rng   = MersenneTwister(42)
    X_raw = rand(rng, 0:1, n_samples, n_features)
    X     = coerce(MLJ.table(X_raw), Count => Continuous)
    y     = randn(rng, n_samples)

    for name in MODEL_NAMES
        GC.gc()
        _, t, bytes, _, _ = @timed tpe_search(name, X, y;
            budget=budget, n_inner_folds=n_inner_folds, seed=42)
        push!(rows, (name, n_samples, n_features, n_inner_folds, budget, t, bytes / 1e9))
        @info "    model=$name" time_s=round(t; digits=3) alloc_gb=round(bytes/1e9; digits=3)
    end
end

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
println()
println("## Data Scaling Benchmark")
println("n_features=$n_features  n_folds=$n_inner_folds  budget=$budget")
println()
@printf("%-16s %10s %10s %10s\n", "model", "n_samples", "time_s", "alloc_GB")
println(repeat("-", 52))
for r in eachrow(rows)
    @printf("%-16s %10d %10.3f %10.3f\n", r.model, r.n_samples, r.time_s, r.alloc_gb)
end

# ---------------------------------------------------------------------------
# Extrapolation note: scale time to budget=50, n_folds=5 using the
# largest measured size as the worst-case baseline.
# ---------------------------------------------------------------------------
println()
println("## Extrapolation to budget=50, n_folds=5 (at largest measured size)")
println()
largest = maximum(SAMPLE_SIZES)
sub = filter(r -> r.n_samples == largest, rows)
scale = (50.0 / budget) * (5.0 / n_inner_folds)
@printf("%-16s %16s %16s\n", "model", "est_s", "est_min")
println(repeat("-", 52))
for r in eachrow(sub)
    est = r.time_s * scale
    @printf("%-16s %16.1f %16.2f\n", r.model, est, est / 60)
end

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
out_dir  = joinpath(@__DIR__, "results")
out_path = joinpath(out_dir, "data_scaling.csv")
mkpath(out_dir)
CSV.write(out_path, rows)
@info "Results written" path=out_path
