#!/usr/bin/env julia
#
# run_ad_annotation.jl — applicability domain (AD) annotation.
#
# Handles two input shard types transparently:
#
#   Layer 1 (from run_model) — one row per (sample_id, run):
#     Averages y_pred over runs, sets is_champion=true (sole model evaluated).
#
#   Model-selection (from run_model_selection) — one row per (sample_id, model):
#     Broadcasts AD scores across all model candidates; is_champion preserved.
#
# For one (dataset, splitter, ratio, rep, fold) split:
#   1. Fit KNNDistanceAD on X_train only (no leakage from test or IPS).
#   2. Score X_test (internal) and X_ips (external) → ad_score, in_domain flag.
#   3. Compute mean nearest-neighbour distance from each test/IPS point to
#      the training set (train–query proximity summary).
#   4. Load the predictions shard and join with AD scores.
#
# Output columns (both paths): dataset, splitter, ratio, rep, fold, set,
#   sample_id, model, is_champion, y_true, y_pred,
#   ad_score, in_domain, nn_dist_to_train
#
# Usage (called by Snakemake):
#   julia --project=. scripts/run_ad_annotation.jl \
#       <dataset> <splitter> <ratio> <rep> <fold> \
#       <split_jld2> <predictions_shard_arrow> <out_arrow>

using DataSplitBench
using Arrow
using DataFrames
using Distances
using JLD2
using MLUtils
using Statistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function load_predictions_shard(path::String, set::String)
    df = DataFrame(Arrow.Table(path))
    filter(r -> r.set == set, df)
end

# Normalise either shard type to one row per (sample_id, model) with is_champion.
# Layer 1 shards have a :run column and no :is_champion; model-selection shards
# have :is_champion and no :run.
function normalise_shard(df::DataFrame)
    if :is_champion in propertynames(df)
        # Model-selection path: already one row per (sample_id, model).
        df[:, [:sample_id, :model, :is_champion, :y_true, :y_pred]]
    else
        # Layer 1 path: average y_pred over independent training runs.
        agg = combine(groupby(df, [:sample_id, :model]),
            :y_true => first => :y_true,
            :y_pred => mean  => :y_pred,
        )
        agg[!, :is_champion] .= true   # sole model in shard is by definition champion
        agg[:, [:sample_id, :model, :is_champion, :y_true, :y_pred]]
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(dataset, splitter, ratio::Float64, rep::Int, fold::Int,
              split_path, predictions_path, out_path)

    roots          = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root
    ad_k  = cfg !== nothing ? cfg.ad.k : 5
    ad_z  = cfg !== nothing ? cfg.ad.z : 3.0

    @info "run_ad_annotation" dataset splitter ratio rep fold

    # Load split indices first (cheap), then build only the needed submatrices.
    sp        = DataSplitBench.load_split(split_path, rep, fold)
    train_idx = sp.train
    test_idx  = sp.test

    data    = DataSplitBench.merck_split_dataloader(data_root, dataset, train_idx, test_idx)
    X_train = data.train.X
    X_test  = data.test.X
    X_ips   = data.ips.X

    # ------------------------------------------------------------------
    # Fit AD on training data only
    # ------------------------------------------------------------------
    @info "Fitting KNNDistanceAD" k=ad_k z=ad_z n_train=size(X_train, 2)
    ad = DataSplitBench.fit_ad(DataSplitBench.KNNDistanceAD, X_train;
                               k=ad_k, z=ad_z, metric=Distances.CosineDist())

    # Score internal test set
    ad_score_test   = DataSplitBench.ad_score(ad, X_test)
    in_domain_test  = DataSplitBench.in_domain(ad, X_test)

    # Score external IPS set
    ad_score_ips    = DataSplitBench.ad_score(ad, X_ips)
    in_domain_ips   = DataSplitBench.in_domain(ad, X_ips)

    # ------------------------------------------------------------------
    # Train–query NN distance (proximity summary, no AD threshold involved)
    # ------------------------------------------------------------------
    nn_dist_test = DataSplitBench.nn_distance(X_train, X_test)
    nn_dist_ips  = DataSplitBench.nn_distance(X_train, X_ips)

    # ------------------------------------------------------------------
    # Load predictions shard and normalise to one row per (sample_id, model)
    # ------------------------------------------------------------------
    pred_test = normalise_shard(load_predictions_shard(predictions_path, "test"))
    pred_ips  = normalise_shard(load_predictions_shard(predictions_path, "ips"))

    @info "Shard type" champion_path=(:is_champion in propertynames(pred_test)) n_models=length(unique(pred_test.model))

    # Build a per-sample AD table, then broadcast across all model rows
    # so each (sample_id, model) row carries the same AD scores.
    function make_shard(sample_ids, ad_scores, in_domain_flags, nn_dists, pred_df, set)
        ad_df = DataFrame(
            sample_id        = collect(Int, sample_ids),
            ad_score         = collect(Float64, ad_scores),
            in_domain        = collect(Bool, in_domain_flags),
            nn_dist_to_train = collect(Float64, nn_dists),
        )
        joined = leftjoin(pred_df, ad_df; on=:sample_id)
        n = nrow(joined)
        hcat(DataFrame(
            dataset  = fill(dataset,  n),
            splitter = fill(splitter, n),
            ratio    = fill(ratio,    n),
            rep      = fill(rep,      n),
            fold     = fill(fold,     n),
            set      = fill(set,      n),
        ), joined)
    end

    shard_test = make_shard(test_idx, ad_score_test, in_domain_test, nn_dist_test, pred_test, "test")
    shard_ips  = make_shard(1:size(X_ips, 2), ad_score_ips, in_domain_ips, nn_dist_ips, pred_ips, "ips")

    result = vcat(shard_test, shard_ips)

    @info "AD coverage" in_domain_test=mean(in_domain_test) in_domain_ips=mean(in_domain_ips)

    mkpath(dirname(out_path))
    Arrow.write(out_path, result; compress=:zstd)

    @info "Done" rows=nrow(result) path=out_path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 8
        println("Usage: run_ad_annotation.jl <dataset> <splitter> <ratio> <rep> <fold> <split_jld2> <predictions_shard_arrow> <out_arrow>")
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
        ARGS[8],
    )
end
