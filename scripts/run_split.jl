#!/usr/bin/env julia

using DataSplitBench
using DataSplits
using Random
using Distances
using JLD2
using Dates

function main(dataset_name, splitter_name, cv_path::String, rep::Int, fold::Int, out_path; frac=0.8, seed=42)
    t0 = Dates.now()

    # Load experiment config when EXPERIMENT_CONFIG is set; otherwise use env-var defaults.
    roots = DataSplitBench.Config.load_experiment_roots()
    cfg, data_root, split_root = roots.cfg, roots.data_root, roots.split_root

    @info "Starting" dataset=dataset_name splitter=splitter_name rep=rep fold=fold frac=frac seed=seed timestamp=t0 data_root=data_root split_root=split_root out_path=out_path

    # When subsampling is configured, cv_path points to the pre-saved subsampled index
    # file written by subsample_cv.jl; otherwise the full outer-train partition is used.
    subsampled = cfg !== nothing && cfg.data.subsample_n !== nothing
    Xsub, ysub, cv_idx = DataSplitBench.load_outer_cv_subset(
        data_root, split_root, dataset_name, rep, fold;
        subsampled_cv_path = subsampled ? cv_path : nothing,
    )
    @info "Outer-train subset loaded" X_size = size(Xsub) y_len = length(ysub) outer_train_len = length(cv_idx)

    # 4) Build splitter via Registry
    splitting_cfg = DataSplitBench.Config.SplittingConfig(
        [splitter_name], [frac],
        cfg !== nothing ? cfg.splitting.repeats : 5,
        cfg !== nothing ? cfg.splitting.k_folds : 5,
    )
    splitter = DataSplitBench.make_splitter(splitter_name, frac, splitting_cfg)
    @info "Splitter ready" splitter = splitter_name frac = frac

    # 5) Split (inner, local indices)
    rng = MersenneTwister(seed)
    res = DataSplitBench.apply_split(Xsub, ysub, splitter; rng=rng)
    @info "Inner split" local_train = length(res.train) local_val = length(res.test)

    # 6) Map back to global indices
    train_idx = cv_idx[res.train]
    val_idx = cv_idx[res.test]
    @info "Mapped to global" train = length(train_idx) val = length(val_idx)

    # 7) Save
    mkpath(dirname(out_path))
    jldopen(out_path, "w"; compress=true) do f
        f["split"] = (train=train_idx, test=val_idx, repeat=rep, fold=fold)
        f["meta"] = Dict(
            "dataset" => dataset_name,
            "splitter" => splitter_name,
            "frac" => frac,
            "seed" => seed,
            "outer_train_len" => length(train_idx),
            "timestamp" => string(Dates.now()),
        )
    end

    elapsed = round(Dates.now() - t0, Dates.Second)
    @info "Done" dataset=dataset_name splitter=splitter_name rep=rep fold=fold path=out_path elapsed=elapsed
    println("Wrote rep=$rep fold=$fold → $out_path  [$(elapsed)]")
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if length(ARGS) < 6
    println("Usage: run_split.jl <dataset> <splitter> <cv_path> <repeat> <fold> <out_jld2> [frac] [seed]")
    exit(1)
end

dataset  = ARGS[1]
splitter = ARGS[2]
cv_path  = ARGS[3]
rep      = parse(Int, ARGS[4])
fold     = parse(Int, ARGS[5])
out_path = ARGS[6]
frac     = length(ARGS) ≥ 7 ? parse(Float64, ARGS[7]) : 0.8
seed     = length(ARGS) ≥ 8 ? parse(Int, ARGS[8]) : 42

main(dataset, splitter, cv_path, rep, fold, out_path; frac=frac, seed=seed)
