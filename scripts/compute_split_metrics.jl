#!/usr/bin/env julia
#
# compute_split_metrics.jl — compute descriptor-space distribution-shift metrics
# for one (dataset, splitter, ratio, rep, fold) split.
#
# Two sets of metrics are computed:
#   internal_* : train  vs. internal test set  (how dissimilar is the held-out test?)
#   external_* : train  vs. external IPS set   (how dissimilar is the benchmark?)
#
# These summaries support the Layer 3 analysis question:
#   "Do splitters with lower estimation error produce internal test sets that
#    better resemble the external benchmark?"
#
# Metrics (from evaluate_split_metrics / SplitMetrics):
#   mahalanobis_distance, sparsity_gap, train_mean, test_mean,
#   train_samples, test_samples, num_features
#   (+ nn_tanimoto_mean, nn_tanimoto_std, js_divergence, coverage_score
#    if data is a BitMatrix — not the case for Merck tabular features)
#
# Usage (called by Snakemake):
#   julia --project=. scripts/compute_split_metrics.jl \
#       <dataset> <splitter> <ratio> <rep> <fold> <split_jld2> <out_arrow>

using DataSplitBench
using Arrow
using DataFrames
using JLD2

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function metrics_to_namedtuple(m, prefix::String)
    d = Dict(m)
    NamedTuple{Tuple(Symbol(prefix * k) for k in keys(d))}(Tuple(values(d)))
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(dataset, splitter, ratio::Float64, rep::Int, fold::Int,
              split_path, out_path)

    roots          = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root

    @info "compute_split_metrics" dataset splitter ratio rep fold

    # Load dataset (features × samples layout)
    data  = DataSplitBench.merck_dataloader(data_root, dataset)
    X_all = data.train.X
    X_ips = data.test.X

    # Load inner split (global indices into X_all) from consolidated splits.jld2
    sp        = DataSplitBench.load_split(split_path, rep, fold)
    train_idx = sp.train
    test_idx  = sp.test

    X_train = X_all[:, train_idx]
    X_test  = X_all[:, test_idx]

    # Internal: train vs test set
    @info "Computing internal shift metrics" n_train=size(X_train,2) n_test=size(X_test,2)
    m_internal = DataSplitBench.evaluate_split_metrics(X_train, X_test)

    # External: train vs IPS (benchmark) set
    @info "Computing external shift metrics" n_train=size(X_train,2) n_ips=size(X_ips,2)
    m_external = DataSplitBench.evaluate_split_metrics(X_train, X_ips)

    # Build single-row DataFrame
    row = DataFrame(
        dataset  = dataset,
        splitter = splitter,
        ratio    = ratio,
        rep      = rep,
        fold     = fold,
    )
    for (key, val) in Dict(m_internal)
        row[!, "internal_" * key] = [val]
    end
    for (key, val) in Dict(m_external)
        row[!, "external_" * key] = [val]
    end

    mkpath(dirname(out_path))
    Arrow.write(out_path, row; compress=:zstd)

    @info "Done" path=out_path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 7
        println("Usage: compute_split_metrics.jl <dataset> <splitter> <ratio> <rep> <fold> <split_jld2> <out_arrow>")
        exit(1)
    end
    main(
        ARGS[1],
        ARGS[2],
        parse(Float64, ARGS[3]),
        parse(Int, ARGS[4]),
        parse(Int, ARGS[5]),
        ARGS[6],
        ARGS[7],
    )
end
