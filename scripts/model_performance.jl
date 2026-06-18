using DataSplitBench
using DataSplits
using Statistics
using StatsBase
using MLJ
using DataFrames
using CSV

LGBMRegressor = MLJ.@load LGBMRegressor pkg = LightGBM

function compute_metrics(y_true, y_pred)
    r2 = RSquared()(y_pred, y_true)
    rmse = RootMeanSquaredError()(y_pred, y_true)
    mae = mean(abs.(y_pred .- y_true))
    medae = median(abs.(y_pred .- y_true))
    mape = MeanAbsoluteProportionalError()(y_pred, y_true)
    pearson = cor(y_pred, y_true)
    spearman = corspearman(y_pred, y_true)
    kendall = corkendall(y_pred, y_true)
    maxerr = maximum(abs.(y_pred .- y_true))
    explained_var = 1 - var(y_pred .- y_true) / var(y_true)
    return Dict(
        "R2" => r2,
        "RMSE" => rmse,
        "MAE" => mae,
        "MedAE" => medae,
        "MAPE" => mape,
        "Pearson" => pearson,
        "Spearman" => spearman,
        "Kendall" => kendall,
        "MaxError" => maxerr,
        "ExplainedVariance" => explained_var
    )
end


function train_and_evaluate(Xt_train, y_train, Xt_test, y_test, Xt_ips, y_ips; seed=nothing)
    model = LGBMRegressor(feature_fraction_seed=seed, feature_fraction=0.8)
    mach = machine(model, Xt_train, y_train)
    fit!(mach, verbosity=0)

    y_pred_test = MLJ.predict(mach, Xt_test)
    metrics_test = compute_metrics(y_test, y_pred_test)

    y_pred_ips = MLJ.predict(mach, Xt_ips)
    metrics_ips = compute_metrics(y_ips, y_pred_ips)

    return Dict("test" => metrics_test, "ips" => metrics_ips)
end


function run_experiment(dataset, algorithm; n_reps=5, n_folds=5, n_runs=10, data_root="Data", split_root="Splits")
    # Load data
    data  = DataSplitBench.merck_dataloader(data_root, dataset)
    X     = data.train.X
    y     = data.train.y
    X_ips = data.test.X
    y_ips = data.test.y
    Xt = MLJ.table(X')
    Xt = coerce(Xt, Count => Continuous)
    Xt_ips = MLJ.table(X_ips')
    Xt_ips = coerce(Xt_ips, Count => Continuous)

    results = DataFrame(
        dataset=String[],
        algorithm=String[],
        rep=Int[],
        fold=Int[],
        run=Int[],
        set=String[],
        R2=Float64[],
        RMSE=Float64[],
        MAE=Float64[],
        MedAE=Float64[],
        MAPE=Float64[],
        Pearson=Float64[],
        Spearman=Float64[],
        Kendall=Float64[],
        MaxError=Float64[],
        ExplainedVariance=Float64[]
    )

    for rep in 1:n_reps
        for fold in 1:n_folds
            split = DataSplitBench.load_method_split(split_root, dataset, algorithm, rep, fold)
            y_train, y_test = DataSplits.splitdata(split, y)
            Xt_train = selectrows(Xt, split.train)
            Xt_test = selectrows(Xt, split.test)

            for run in 1:n_runs
                seed = 10000 * rep + 100 * fold + run
                metrics = train_and_evaluate(Xt_train, y_train, Xt_test, y_test, Xt_ips, y_ips; seed=seed)
                for set in ["test", "ips"]
                    row = (
                        dataset,
                        algorithm,
                        rep,
                        fold,
                        run,
                        set,
                        metrics[set]["R2"],
                        metrics[set]["RMSE"],
                        metrics[set]["MAE"],
                        metrics[set]["MedAE"],
                        metrics[set]["MAPE"],
                        metrics[set]["Pearson"],
                        metrics[set]["Spearman"],
                        metrics[set]["Kendall"],
                        metrics[set]["MaxError"],
                        metrics[set]["ExplainedVariance"]
                    )
                    push!(results, row)
                end
            end
        end
    end
    return results
end

function main(dataset, algorithm)
    # Load experiment config when EXPERIMENT_CONFIG is set; otherwise use defaults.
    roots = DataSplitBench.Config.load_experiment_roots(data_default="Data/merck")
    cfg, data_root = roots.cfg, roots.data_root
    split_root = cfg !== nothing ? DataSplitBench.Config.splits_root(cfg) : "Splits/merck"
    n_reps = cfg !== nothing ? cfg.splitting.repeats : 5
    n_folds = cfg !== nothing ? cfg.splitting.k_folds : 5
    outdir = cfg !== nothing ? joinpath(DataSplitBench.Config.model_results_root(cfg), dataset, algorithm) : joinpath("results", dataset, algorithm)

    results = run_experiment(dataset, algorithm;
        n_reps=n_reps, n_folds=n_folds, n_runs=10,
        data_root=data_root, split_root=split_root)

    mkpath(outdir)
    CSV.write(joinpath(outdir, "metrics.csv"), results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        error("Usage: julia scripts/model_performance.jl <dataset> <algorithm>")
    end
    main(ARGS[1], ARGS[2])
end
