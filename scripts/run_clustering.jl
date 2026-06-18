#!/usr/bin/env julia

"""
run_cluster_cv.jl — cluster OUTER-TRAIN data using Clustering.jl, optionally after UMAP

Usage:
  run_cluster_cv.jl <dataset> <algorithm> <cv_root> <repeat> <fold> <out_prefix> <k> [seed] [linkage] [metric] [umap_dim] [umap_neighbors] [umap_min_dist] [umap_metric]

Examples:
  # Plain clustering (as before)
  run_cluster_cv.jl "dataset_123" kmeans  "Splits/merck" 1 0 "out/merck123_kmeans_rep1_fold0" 20 42

  # UMAP + KMeans
  run_cluster_cv.jl "dataset_123" umap-kmeans "Splits/merck" 1 0 "out/merck123_umapkmeans_rep1_fold0" 20 42 "ward" "euclidean" 10 15 0.1 "euclidean"

  # UMAP + HAC (Ward on UMAP space)
  run_cluster_cv.jl "dataset_123" umap-hac "Splits/merck" 1 0 "out/merck123_umaphac_rep1_fold0" 20 42 "ward" "euclidean" 10 15 0.1 "euclidean"

Outputs:
  <out_prefix>_assignments.csv  # global_index, cluster, rep, fold
  <out_prefix>_centers.csv      # if kmeans
  <out_prefix>_medoids.csv      # if kmedoids
  <out_prefix>_umap.csv         # if UMAP used: global_index, comp_1..comp_d
  <out_prefix>.jld2             # rich data (assignments, indices, algo meta, and UMAP if used)
"""

using DataSplitBench
using DataSplits
using Random
using Distances
using JLD2
using Dates
using Clustering
using CSV
using DataFrames
using Statistics
using UMAP

###
# Folder Structure #
###

# Clustering
# # merck
#   # dataset
#     # Clustering-algorithm
#       # rep{n}_fold{n}.jld2

# ---------------------------
# Helpers
# ---------------------------


# ---------------------------
# Clustering runners
# ---------------------------

struct ClusterResult
    assignments::Vector{Int}
    centers::Union{Nothing,Matrix}
    medoid_local_idx::Union{Nothing,Vector{Int}}
    hac_merge_heights::Union{Nothing,Vector{Float64}}
end

function run_kmeans(data_fxs::AbstractMatrix, k::Int; metric=Distances.Euclidean(), rng, maxiter::Int=300)
    res = Clustering.kmeans(data_fxs, k; distance=metric, init=:kmpp, maxiter=maxiter, display=:none, rng=rng)
    return ClusterResult(res.assignments, res.centers, nothing, nothing)
end

function run_kmedoids(data_fxs::AbstractMatrix, k::Int; metric=Distances.Euclidean(), rng)
    D = Distances.pairwise(metric, data_fxs)
    res = Clustering.kmedoids(D, k; display=:none)
    return ClusterResult(res.assignments, nothing, res.medoids, nothing)
end

function run_butina(data_fxs::AbstractMatrix; metric=Distances.Euclidean())
    res = DataSplits.sphere_exclusion(data_fxs; radius=0.35, metric=metric)
    return ClusterResult(res.assignments, nothing, nothing, nothing)
end

function run_hac(data_fxs::AbstractMatrix; k::Int=50, linkage::Symbol=:ward, metric=Distances.Euclidean())
    D = pairwise(metric, data_fxs; dims=2)  # distances between columns (samples)
    tree = Clustering.hclust(D, linkage)
    assigns = Clustering.cutree(tree; k=k)
    return ClusterResult(assigns, nothing, nothing, tree.heights)
end

# ---------------------------
# UMAP
# ---------------------------

"""
Run UMAP on (features x samples). Returns:
- U_fx_s: (umap_dim x samples) for clustering
"""
function run_umap(data_fxs::AbstractMatrix; umap_dim::Int=10, umap_neighbors::Int=15, umap_min_dist::Float64=0.1, umap_metric=:euclidean)
    data_fxs = Float32.(data_fxs)
    U_sxd = UMAP.umap(data_fxs, umap_dim; n_neighbors=umap_neighbors, min_dist=umap_min_dist, metric=umap_metric)
    return U_sxd
end

# ---------------------------
# Main
# ---------------------------

function main(dataset_name, algorithm_name, cv_root, rep::Int, fold::Int, out_prefix;
    k::Int=4, seed::Int=42, linkage_name::AbstractString="ward", metric_name::AbstractString="euclidean",
    umap_dim::Int=10, umap_neighbors::Int=15, umap_min_dist::Float64=0.1)

    roots = DataSplitBench.Config.load_experiment_roots()
    data_root = roots.data_root
    @info "Params" dataset = dataset_name algorithm = algorithm_name rep = rep fold = fold k = k seed = seed data_root = data_root cv_root = cv_root out_prefix = out_prefix linkage = linkage_name metric = metric_name umap_dim = umap_dim umap_neighbors = umap_neighbors umap_min_dist = umap_min_dist

    # 1–2) Load data, outer CV fold, and subset to outer-train (cv_root from CLI arg)
    Xsub, _, cv_idx = DataSplitBench.load_outer_cv_subset(data_root, cv_root, dataset_name, rep, fold)
    @info "Outer-train subset loaded" Xsub_size = size(Xsub) n_outer_train = length(cv_idx) X_eltype = eltype(Xsub)

    n_features, n_samples = size(Xsub)
    @info "Input data" features = n_features samples = n_samples

    rng = MersenneTwister(seed)
    alg = lowercase(strip(algorithm_name))
    metric = DataSplitBench.parse_distance_metric(metric_name)
    linkage = DataSplitBench.parse_linkage(linkage_name)

    # Detect UMAP-prefixed algorithms
    use_umap = startswith(alg, "umap")
    base_alg = use_umap ? replace(alg, "umap-" => "", "umap_" => "", "umap" => "kmeans") : alg  # default to kmeans if just "umap"

    # 4) UMAP (optional)
    U_fx_s = nothing
    if use_umap
        @info "Running UMAP" dim = umap_dim neighbors = umap_neighbors min_dist = umap_min_dist metric = metric
        U_fx_s = run_umap(Xsub; umap_dim=umap_dim, umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist, umap_metric=metric)
        @info "UMAP done" umap_shape = (size(U_fx_s, 1), size(U_fx_s, 2))
    end

    data_for_cluster = use_umap ? U_fx_s : Xsub

    # 5) Cluster
    @info "Clustering" algorithm = base_alg seed = seed metric = string(typeof(metric)) linkage = string(linkage)
    cres = if base_alg == "kmeans"
        run_kmeans(data_for_cluster, k; metric=metric, rng=rng)
    elseif base_alg == "kmedoids"
        run_kmedoids(data_for_cluster, k; metric=metric, rng=rng)
    elseif base_alg == "hac" || base_alg == "hierarchical" || base_alg == "hclust"
        run_hac(data_for_cluster; k=50, linkage=linkage, metric=metric)
    elseif base_alg == "butina"
        run_butina(data_for_cluster, metric=metric)
    else
        error("Unknown algorithm: $algorithm_name (supported: kmeans, kmedoids, hac, and umap-* variants)")
    end

    # 6) Save results
    mkpath(dirname(out_prefix))
    global_idx = collect(cv_idx)
    assignments = cres.assignments
    @assert length(assignments) == length(cv_idx)
    ts = string(Dates.now())

    # Assignments
    df = DataFrame(
        global_index=global_idx,
        cluster=assignments,
        rep=fill(rep, length(assignments)),
        fold=fill(fold, length(assignments))
    )
    csv_assign = out_prefix * "_assignments.csv"
    CSV.write(csv_assign, df)

    # UMAP CSV (if used): rows = samples, columns = comp_1..comp_d
    csv_umap = nothing
    if use_umap
        comps = size(U_fx_s, 1)
        dfu = DataFrame(global_index=global_idx)
        for j in 1:comps
            dfu[!, "comp_$j"] = U_fx_s[j, :]
        end
        csv_umap = out_prefix * "_umap.csv"
        CSV.write(csv_umap, dfu)
    end

    # JLD2 bundle
    out_jld2 = out_prefix * ".jld2"
    jldopen(out_jld2, "w"; compress=true) do f
        f["clusters"] = (assignments=assignments, rep=rep, fold=fold, k=k)
        f["indices"] = (global_idx,)
        f["algo"] = Dict(
            "name" => base_alg,
            "seed" => seed,
            "metric" => metric_name,
            "linkage" => string(linkage),
            "used_umap" => use_umap,
        )
        if cres.centers !== nothing
            f["kmeans"] = (centers=cres.centers,)
        end
        if cres.medoid_local_idx !== nothing
            f["kmedoids"] = (medoid_local=cres.medoid_local_idx,
                medoid_global=global_idx[cres.medoid_local_idx],)
        end
        if cres.hac_merge_heights !== nothing
            f["hac"] = (merge_heights=cres.hac_merge_heights,)
        end
        if use_umap
            f["umap"] = Dict(
                "embedding_fx_s" => U_fx_s,          # dim x samples (for direct clustering reuse)
                "dim" => size(U_fx_s, 1),
                "n_neighbors" => umap_neighbors,
                "min_dist" => umap_min_dist,
                "metric" => metric_name,
            )
        end
        f["meta"] = Dict(
            "dataset" => dataset_name,
            "n_features" => n_features,
            "n_samples_outer_train" => n_samples,
            "timestamp" => ts
        )
    end

    @info "Saved" assignments_csv = csv_assign jld2 = out_jld2 umap_csv = csv_umap
    println("Wrote clustering results →")
    println("  CSV assignments: $csv_assign")
    if csv_umap !== nothing
        println("  CSV UMAP       : $(csv_umap)")
    end
    println("  JLD2           : $out_jld2")
end

# ---------------------------
# CLI
# ---------------------------

if length(ARGS) < 7
    println("Usage: run_cluster_cv.jl <dataset> <algorithm> <cv_root> <repeat> <fold> <out_prefix> [metric]")
    exit(1)
end

dataset = ARGS[1]
algorithm = ARGS[2]         # kmeans | kmedoids | hac | umap-kmeans | umap-kmedoids | umap-hac
cv_root = ARGS[3]
rep = parse(Int, ARGS[4])
fold = parse(Int, ARGS[5])
out_prefix = ARGS[6]
metric_name = length(ARGS) ≥ 7 ? ARGS[7] : "euclidean"


main(dataset, algorithm, cv_root, rep, fold, out_prefix; metric_name=metric_name)
