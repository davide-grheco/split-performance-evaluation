#!/usr/bin/env julia
#
# run_hyperopt.jl — train one model with TPE hyperparameter optimisation on one
# (dataset, splitter, ratio, rep, fold) split and write a prediction shard.
#
# Workflow:
#   1. Load the pre-computed train/test split (JLD2)
#   2. Run TPE search on the train set via tpe_search()
#   3. Best model is retrained on the full train set inside tpe_search()
#   4. Evaluate on the inner test set and the external IPS set
#   5. Write an Arrow shard with predictions + best hyperparams + CV score
#
# Usage (called by Snakemake):
#   julia --project=. scripts/run_hyperopt.jl \
#       <dataset> <splitter> <ratio> <model> <rep> <fold> \
#       <split_jld2> <out_arrow>

using DataSplitBench
using Arrow
using DataFrames
using JLD2
using MLJ
using MLUtils
using JSON3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function hyperopt_shard_df(sample_ids, y_true, y_pred,
                            best_params_json, best_cv_rms,
                            set, dataset, splitter, ratio, model_name, rep, fold)
    n = length(sample_ids)
    DataFrame(
        dataset        = fill(dataset,          n),
        splitter       = fill(splitter,         n),
        ratio          = fill(ratio,            n),
        model          = fill(model_name,       n),
        rep            = fill(rep,              n),
        fold           = fill(fold,             n),
        set            = fill(set,              n),
        best_params    = fill(best_params_json, n),
        best_cv_rms    = fill(best_cv_rms,      n),
        sample_id      = collect(sample_ids),
        y_true         = collect(Float64, y_true),
        y_pred         = collect(Float64, y_pred),
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(dataset, splitter, ratio::Float64, model_name, rep::Int, fold::Int,
              split_path, out_path)

    roots = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root
    n_inner_folds = cfg !== nothing ? cfg.splitting.k_folds : 5
    tpe_budget    = cfg !== nothing ? cfg.models.tpe_budget  : 50

    @info "run_hyperopt" dataset splitter ratio model_name rep fold split_path

    # Load full dataset
    data  = DataSplitBench.merck_dataloader(data_root, dataset)
    X_all = data.train.X
    y_all = data.train.y
    X_ips = data.test.X
    y_ips = data.test.y
    Xt_all = coerce(MLJ.table(X_all'), Count => Continuous)
    Xt_ips = coerce(MLJ.table(X_ips'), Count => Continuous)

    # Load inner split (global indices into X_all)
    sp        = DataSplitBench.load_split(split_path)
    train_idx = sp.train
    test_idx  = sp.test

    Xt_train = selectrows(Xt_all, train_idx)
    Xt_test  = selectrows(Xt_all, test_idx)
    y_train  = y_all[train_idx]
    y_test   = y_all[test_idx]

    # TPE search: inner CV on train, returns best params + retrained machine
    seed = Int(abs(hash((dataset, splitter, ratio, model_name, rep, fold))) % typemax(Int32))
    best_params, best_cv_rms, mach = DataSplitBench.tpe_search(
        model_name, Xt_train, y_train;
        budget=tpe_budget, n_inner_folds=n_inner_folds, seed=seed,
    )
    best_params_json = JSON3.write(best_params)

    @info "Best params" best_params best_cv_rms

    # Predict on internal test and external IPS sets
    y_pred_test = MLJ.predict(mach, Xt_test)
    y_pred_ips  = MLJ.predict(mach, Xt_ips)

    shards = [
        hyperopt_shard_df(test_idx, y_test, y_pred_test,
                          best_params_json, best_cv_rms,
                          "test", dataset, splitter, ratio, model_name, rep, fold),
        hyperopt_shard_df(1:length(y_ips), y_ips, y_pred_ips,
                          best_params_json, best_cv_rms,
                          "ips", dataset, splitter, ratio, model_name, rep, fold),
    ]

    result = vcat(shards...)
    mkpath(dirname(out_path))
    Arrow.write(out_path, result; compress=:zstd)

    @info "Done" rows=nrow(result) path=out_path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 8
        println("Usage: run_hyperopt.jl <dataset> <splitter> <ratio> <model> <rep> <fold> <split_jld2> <out_arrow>")
        exit(1)
    end
    main(
        ARGS[1],
        ARGS[2],
        parse(Float64, ARGS[3]),
        ARGS[4],
        parse(Int, ARGS[5]),
        parse(Int, ARGS[6]),
        ARGS[7],
        ARGS[8],
    )
end
