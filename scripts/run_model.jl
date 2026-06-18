#!/usr/bin/env julia
#
# run_model.jl — train one model on one (dataset, splitter, ratio, rep, fold)
# and write raw predictions as an Arrow shard.
#
# Usage (called by Snakemake):
#   julia --project=. scripts/run_model.jl \
#       <dataset> <splitter> <ratio> <model> <rep> <fold> \
#       <split_jld2> <out_arrow>

using DataSplitBench
using Arrow
using DataFrames
using Dates
using JLD2
using MLJ
using MLUtils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function predictions_df(sample_ids, y_true, y_pred, set::String,
                        dataset, splitter, ratio, model_name, rep, fold, run)
    n = length(sample_ids)
    DataFrame(
        dataset   = fill(dataset,    n),
        splitter  = fill(splitter,   n),
        ratio     = fill(ratio,      n),
        model     = fill(model_name, n),
        rep       = fill(rep,        n),
        fold      = fill(fold,       n),
        run       = fill(run,        n),
        set       = fill(set,        n),
        sample_id = collect(sample_ids),
        y_true    = collect(Float64, y_true),
        y_pred    = collect(Float64, y_pred),
    )
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(dataset, splitter, ratio::Float64, model_name, rep::Int, fold::Int,
              split_path, out_path)

    t0 = Dates.now()
    roots  = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root
    n_runs = cfg !== nothing ? cfg.models.n_runs : 10

    @info "Starting" dataset splitter ratio model_name rep fold n_runs timestamp=t0

    # Load full dataset (X_all = training corpus, X_ips = external test set)
    data  = DataSplitBench.merck_dataloader(data_root, dataset)
    X_all = data.train.X
    y_all = data.train.y
    X_ips = data.test.X
    y_ips = data.test.y
    Xt_all = coerce(MLJ.table(X_all'), Count => Continuous)
    Xt_ips = coerce(MLJ.table(X_ips'), Count => Continuous)

    # Load inner split (global indices into X_all) from consolidated splits.jld2
    sp        = DataSplitBench.load_split(split_path, rep, fold)
    train_idx = sp.train
    test_idx  = sp.test

    Xt_train = selectrows(Xt_all, train_idx)
    Xt_test  = selectrows(Xt_all, test_idx)
    y_train  = y_all[train_idx]
    y_test   = y_all[test_idx]

    # Train all runs in parallel. Each run is seeded differently so results are
    # independent. shards is pre-allocated to avoid thread-unsafe push!.
    shards = Vector{DataFrame}(undef, 2 * n_runs)

    Threads.@threads for run in 1:n_runs
        t_run = Dates.now()
        @info "Run started" run n_runs timestamp=t_run

        seed  = Int(abs(hash((dataset, splitter, ratio, model_name, rep, fold, run))) % typemax(Int32))
        model = DataSplitBench.make_model(model_name, cfg.models; seed=seed)
        mach  = machine(model, Xt_train, y_train)
        fit!(mach; verbosity=0)

        shards[2*run-1] = predictions_df(
            test_idx, y_test, MLJ.predict(mach, Xt_test),
            "test", dataset, splitter, ratio, model_name, rep, fold, run,
        )
        shards[2*run] = predictions_df(
            1:length(y_ips), y_ips, MLJ.predict(mach, Xt_ips),
            "ips", dataset, splitter, ratio, model_name, rep, fold, run,
        )

        elapsed_run = round(Dates.now() - t_run, Dates.Second)
        @info "Run done" run n_runs elapsed=elapsed_run
    end

    result = vcat(shards...)
    mkpath(dirname(out_path))
    Arrow.write(out_path, result; compress=:zstd)

    elapsed = round(Dates.now() - t0, Dates.Second)
    @info "Done" rows=nrow(result) path=out_path elapsed=elapsed
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 8
        println("Usage: run_model.jl <dataset> <splitter> <ratio> <model> <rep> <fold> <split_jld2> <out_arrow>")
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
