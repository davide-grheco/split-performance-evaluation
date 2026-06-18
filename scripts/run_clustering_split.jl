using CSV
using DataFrames
using Clustering
using StatsBase
using DataSplits
using DataSplitBench
using Random
using JLD2
using Dates

struct Cluster <: Clustering.ClusteringResult
    assignments::Vector{Int}
    counts::Vector{Int}
end


function load_clustering(base_folder, dataset_name, clustering_algorithm, distance_metric, rep, fold)

    assignments_csv = joinpath(base_folder, dataset_name, clustering_algorithm, distance_metric, "rep$(rep)_fold$(fold)_assignments.csv")
    clusters = CSV.read(assignments_csv, DataFrame)

    counts = counts_from_assignments(clusters[!, :cluster])
    assignments = Cluster(clusters[!, :cluster], counts)
    return assignments
end


function main(dataset_name, splitter_name, clustering_algorithm, distance_metric, rep::Int, fold::Int, out_path; frac=0.8, seed=42)

    # 1) Load data
    roots = DataSplitBench.Config.load_experiment_roots()
    cfg, data_root, split_root = roots.cfg, roots.data_root, roots.split_root
    cluster_root = get(ENV, "CLUSTER_ROOT", "Clustering/merck/")
    @info "Params" dataset = dataset_name splitter = splitter_name clustering_algorithm = clustering_algorithm distance_metric = distance_metric rep = rep fold = fold frac = frac seed = seed data_root = data_root split_root = split_root out_path = out_path

    X, y, cv_idx = DataSplitBench.load_outer_cv_subset(data_root, split_root, dataset_name, rep, fold)

    clusters = load_clustering(cluster_root, dataset_name, clustering_algorithm, distance_metric, rep, fold)

    rng = MersenneTwister(seed)

    splitter = DataSplitBench.make_cluster_splitter(clusters, splitter_name; frac=frac)
    @info "Splitter ready" splitter = splitter_name splitter_type = string(typeof(splitter)) frac = frac

    @info "Splitting" method = splitter_name seed = seed
    res = DataSplits.split(X, splitter)

    @info "Inner split (local)" local_train_len = length(res.train) local_val_len = length(res.test)

    # 6) Map back to GLOBAL indices
    train_idx = cv_idx[res.train]
    val_idx = cv_idx[res.test]
    @info "Mapped to global" train_len = length(train_idx) val_len = length(val_idx) min_train = minimum(train_idx) max_train = maximum(train_idx)

    @info "Saving JLD2" path = out_path

    # 7) Save to JLD2 (inner split only; we discard the outer test on purpose)
    mkpath(dirname(out_path))
    jldopen(out_path, "w"; compress=true) do f
        f["split"] = (train=train_idx,
            test=val_idx,
            repeat=rep,
            fold=fold)
        f["meta"] = Dict(
            "dataset" => dataset_name,
            "splitter" => splitter_name,
            "clustering" => clustering_algorithm,
            "distance_metric" => distance_metric,
            "frac" => frac,
            "seed" => seed,
            "outer_train_len" => length(train_idx),
            "timestamp" => string(Dates.now())
        )
    end


    @info "Done" dataset = dataset_name splitter = splitter_name rep = rep fold = fold path = out_path
    println("Wrote rep=$rep fold=$fold → $out_path")
end


# ---------------------------
# CLI
# ---------------------------
if length(ARGS) < 6
    println("Usage: run_split_cv.jl <dataset> <splitter> <clustering> <distance> <repeat> <fold> <out_jld2> [frac] [seed]")
    exit(1)
end

dataset = ARGS[1]
splitter = ARGS[2]
clustering = ARGS[3]
distance = ARGS[4]
rep = parse(Int, ARGS[5])
fold = parse(Int, ARGS[6])
out_path = ARGS[7]
frac = length(ARGS) ≥ 8 ? parse(Float64, ARGS[8]) : 0.8
seed = length(ARGS) ≥ 9 ? parse(Int, ARGS[9]) : 42

main(dataset, splitter, clustering, distance, rep, fold, out_path; frac=frac, seed=seed)
