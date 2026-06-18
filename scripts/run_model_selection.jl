#!/usr/bin/env julia
#
# run_model_selection.jl — champion model selection via successive-halving TPE.
#
# For one (dataset, splitter, ratio, rep, fold) split:
#   1. Run successive-halving TPE across all model candidates.
#      Round 1: all models, small budget, 2 inner folds  → keep top 2
#      Round 2: top 2 models, larger budget, full folds  → keep top 1
#      Round 3: champion only, remaining budget           → best config
#   2. Retrain ALL candidates on the full training set.
#   3. Evaluate every candidate on the internal test set and the external IPS set.
#   4. Write one Arrow shard with predictions + metadata for every candidate,
#      flagging the champion.  This enables oracle-regret analysis at gather time.
#
# Usage (called by Snakemake):
#   julia --project=. scripts/run_model_selection.jl \
#       <dataset> <splitter> <ratio> <rep> <fold> \
#       <split_jld2> <out_arrow>

using DataSplitBench
using Arrow
using DataFrames
using Dates
using JLD2
using MLJ
using MLUtils
using JSON3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function candidate_shard_df(sample_ids, y_true, y_pred,
                             result::DataSplitBench.CandidateResult,
                             set, dataset, splitter, ratio, rep, fold)
    n = length(sample_ids)
    DataFrame(
        dataset          = fill(dataset,                  n),
        splitter         = fill(splitter,                 n),
        ratio            = fill(ratio,                    n),
        rep              = fill(rep,                      n),
        fold             = fill(fold,                     n),
        set              = fill(set,                      n),
        model            = fill(result.name,              n),
        is_champion      = fill(result.is_champion,       n),
        round_eliminated = fill(result.round_eliminated,  n),
        best_params      = fill(JSON3.write(result.params), n),
        cv_rms           = fill(result.cv_rms,            n),
        sample_id        = collect(sample_ids),
        y_true           = collect(Float64, y_true),
        y_pred           = collect(Float64, y_pred),
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(dataset, splitter, ratio::Float64, rep::Int, fold::Int,
              split_path, out_path)

    roots         = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root
    n_inner_folds    = cfg !== nothing ? cfg.splitting.k_folds : 5
    tpe_budget       = cfg !== nothing ? (cfg.layer2 !== nothing ? cfg.layer2.tpe_budget       : cfg.models.tpe_budget) : 50
    model_names      = cfg !== nothing ? (cfg.layer2 !== nothing ? cfg.layer2.model_candidates  : cfg.models.methods)   : ["lightgbm"]
    tpe_subsample_n  = cfg !== nothing && cfg.layer2 !== nothing ? cfg.layer2.tpe_subsample_n : nothing

    t0 = Dates.now()
    @info "Starting" dataset splitter ratio rep fold n_candidates=length(model_names) tpe_budget timestamp=t0

    # Load full dataset
    data  = DataSplitBench.merck_dataloader(data_root, dataset)
    X_all = data.train.X
    y_all = data.train.y
    X_ips = data.test.X
    y_ips = data.test.y
    Xt_all = coerce(MLJ.table(X_all'), Count => Continuous)
    Xt_ips = coerce(MLJ.table(X_ips'), Count => Continuous)

    # Load inner split from consolidated splits.jld2
    sp        = DataSplitBench.load_split(split_path, rep, fold)
    train_idx = sp.train
    test_idx  = sp.test

    Xt_train = selectrows(Xt_all, train_idx)
    Xt_test  = selectrows(Xt_all, test_idx)
    y_train  = y_all[train_idx]
    y_test   = y_all[test_idx]

    # Successive-halving TPE: returns one CandidateResult per model
    t_tpe = Dates.now()
    @info "TPE started" timestamp=t_tpe
    base_seed = Int(abs(hash((dataset, splitter, ratio, rep, fold))) % typemax(Int32))
    candidates = DataSplitBench.select_champion(model_names, Xt_train, y_train;
                                                budget=tpe_budget,
                                                n_inner_folds=n_inner_folds,
                                                base_seed=base_seed,
                                                tpe_subsample_n=tpe_subsample_n)

    champion = candidates[findfirst(c -> c.is_champion, candidates)]
    elapsed_tpe = round(Dates.now() - t_tpe, Dates.Second)
    @info "Champion selected" champion=champion.name cv_rms=champion.cv_rms elapsed_tpe=elapsed_tpe

    # Retrain one candidate at a time and collect predictions before retraining
    # the next, so only one fitted machine is live at a time.
    shards = DataFrame[]
    for result in candidates
        mach        = DataSplitBench.retrain_candidate(result, Xt_train, y_train, base_seed)
        y_pred_test = MLJ.predict(mach, Xt_test)
        y_pred_ips  = MLJ.predict(mach, Xt_ips)

        push!(shards, candidate_shard_df(
            test_idx, y_test, y_pred_test,
            result, "test", dataset, splitter, ratio, rep, fold,
        ))
        push!(shards, candidate_shard_df(
            1:length(y_ips), y_ips, y_pred_ips,
            result, "ips", dataset, splitter, ratio, rep, fold,
        ))
        GC.gc(true)
    end

    result_df = vcat(shards...)
    mkpath(dirname(out_path))
    Arrow.write(out_path, result_df; compress=:zstd)

    elapsed = round(Dates.now() - t0, Dates.Second)
    @info "Done" rows=nrow(result_df) n_candidates=length(candidates) path=out_path elapsed=elapsed
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 7
        println("Usage: run_model_selection.jl <dataset> <splitter> <ratio> <rep> <fold> <split_jld2> <out_arrow>")
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
