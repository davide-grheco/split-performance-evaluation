using Random
import Clustering: ClusteringResult, assignments, counts, nclusters
using Distances
using Combinatorics
using MLUtils
import Base: ==, hash

export ButinaResult, butina, merge_small_clusters

"""
    ButinaResult <: ClusteringResult

Result of Butina clustering.

# Fields
- `clusters::Vector{Vector{Int}}` — the raw clusters as lists of indices
- `assignments::Vector{Int}`     — for each point, the cluster‐ID (1..k)
- `counts::Vector{Int}`          — number of points in each cluster
- `cutoff::Real`                 — Tanimoto cutoff used
"""
struct ButinaResult <: ClusteringResult
    assignments::Vector{Int}
    counts::Vector{Int}
    cutoff::Real
end

Base.:(==)(a::ButinaResult, b::ButinaResult) =
    a.assignments == b.assignments &&
    a.counts == b.counts &&
    a.cutoff == b.cutoff

Base.hash(br::ButinaResult, h::UInt) = hash((br.assignments, br.counts, br.cutoff), h)

nclusters(r::ButinaResult) = length(counts(r))
assignments(r::ButinaResult) = r.assignments
counts(r::ButinaResult) = r.counts
"""
    butina(data::AbstractVector{T};
           threshold::Real=0.2,
           metric::SemiMetric=Jaccard(),
           reorder::Bool=true) → ButinaResult

Perform Butina clustering (sphere exclusion algorithm) on a dataset. This is particularly useful for molecular datasets where
we want to select diverse representatives.

### Algorithm Overview
1. **Neighbor Identification**: Find all elements within `threshold` distance using `metric`
2. **Centroid Selection**:
   - Without reordering: Sort by descending neighbor count
   - With reordering (`reorder=true`): Always select element with most unassigned neighbors
3. **Cluster Growth**: Form clusters by assigning unassigned neighbors to centroids

### Key Features
- Handles both dense and sparse datasets efficiently
- Optional dynamic reordering for better cluster quality
- Preserves original indices in output

### Arguments
- `data`       : Input collection (typically fingerprints or feature vectors)
- `threshold`  : Maximum distance for elements to be considered neighbors (0.0-1.0 for similarity metrics)
- `metric`     : Distance metric (must satisfy triangle inequality). Common choices:
                 - `Tanimoto()`/`Jaccard()` for fingerprints
                 - `Euclidean()` for continuous features
- `reorder`    : If `true`, dynamically updates neighbor counts during clustering (recommended)

### Returns
A `ButinaResult` struct containing:
- `assignments` : Vector of cluster IDs for each input element
- `counts`      : Number of elements in each cluster
- `threshold`   : The distance threshold used
"""
function butina(data::AbstractVector; threshold::Real=0.2, metric::SemiMetric=Jaccard(), reorder::Bool=true)
    n = numobs(data)
    @assert 0 <= threshold <= 1 "cutoff must be between 0 and 1"

    result = neighbour_lists(data, threshold=threshold, metric=metric)
    neighbors, original_indices = result.lists, result.map

    assigned = falses(n)
    clusters = Vector{Vector{Int}}()
    current_counts = length.(neighbors)

    while any(!, assigned)
        unassigned = findall(!, assigned)
        candidates = [(i, current_counts[i]) for i in unassigned]

        max_count = maximum(c -> c[2], candidates)
        max_candidates = [c[1] for c in candidates if c[2] == max_count]
        next_centroid = max_candidates[1]

        cluster_members = [next_centroid]
        assigned[next_centroid] = true
        for neighbor in neighbors[next_centroid]
            if !assigned[neighbor]
                push!(cluster_members, neighbor)
                assigned[neighbor] = true
            end
        end

        push!(clusters, cluster_members)

        if reorder
            for member in cluster_members
                for neighbor in neighbors[member]
                    current_counts[neighbor] -= 1
                end
            end
        end
    end

    original_clusters = [original_indices[cluster] for cluster in clusters]
    counts = length.(original_clusters)

    assignments = zeros(Int, n)
    for (cluster_id, cluster) in enumerate(clusters)
        assignments[original_indices[cluster]] .= cluster_id
    end

    return ButinaResult(assignments, counts, threshold)

end

function butina(X::BitMatrix; threshold::Real=0.2, metric::SemiMetric=Jaccard(), reorder::Bool=true)
    data_vec = collect(eachobs(X))
    return butina(data_vec; threshold=threshold, metric=metric, reorder=reorder)
end

function butina(X::AbstractMatrix; threshold::Real=0.2, metric::SemiMetric=Jaccard(), reorder::Bool=true)
    data_vec = collect(eachobs(X))
    return butina(data_vec; threshold=threshold, metric=metric, reorder=reorder)
end

function clusters_from_assignments(assignments::AbstractVector{Int}, k::Int=maximum(assignments))
    clusters = [Int[] for _ in 1:k]
    for (idx, cid) in pairs(assignments)
        push!(clusters[cid], idx)
    end
    return clusters
end

function counts_from_assignments(assignments::AbstractVector{<:Integer})
    isempty(assignments) && return Int[]
    k = maximum(assignments)
    counts = zeros(Int, k)
    for cid in assignments
        cid >= 1 || throw(ArgumentError("cluster assignments must be positive integers"))
        counts[cid] += 1
    end
    return counts
end

clusters(result::ClusteringResult) = clusters_from_assignments(assignments(result), nclusters(result))

function point_to_cluster_distance(data, idx, cluster, metric)
    xi = getobs(data, idx)
    minimum(Distances.evaluate(metric, xi, getobs(data, j)) for j in cluster)
end

"""
    merge_small_clusters(butina_result::ButinaResult, data::AbstractVector, cluster_size_threshold::Int; metric=Euclidean())

Merge small clusters from Butina clustering into their nearest large clusters.

# Arguments
- `butina_result::ButinaResult`: Result from Butina clustering containing cluster assignments and counts
- `data::AbstractVector`: The original data points that were clustered
- `cluster_size_threshold::Int`: Minimum size required for a cluster to be considered "large" (not merged)

# Keyword Arguments
- `metric=Euclidean()`: Distance metric used for determining nearest clusters (default: Euclidean distance)

# Returns
- `ButinaResult`: A new clustering result where:
  - All clusters smaller than `cluster_size_threshold` have been merged into their nearest large cluster
  - Large clusters retain their original members plus any merged small clusters
  - Cluster assignments and counts are updated accordingly

# Description
This function processes the results of Butina clustering by:
1. Identifying all clusters smaller than the specified threshold
2. For each small cluster, finding the nearest large cluster based on the specified distance metric
3. Merging the small cluster members into their nearest large cluster
4. Returning updated cluster assignments and counts

The distance between a small cluster member and a large cluster is calculated as the minimum distance between that member and any point in the large cluster.

This algorithm was implemented following the Description from:
Landrum GA, Beckers M, Lanini J, Schneider N, Stiefl N, Riniker S.
SIMPD: an algorithm for generating simulated time splits for validating machine learning approaches. J Cheminform. 2023 Dec 11;15(1):119.
"""
function merge_small_clusters(butina_result::ClusteringResult, data::AbstractVector, cluster_size_threshold::Int; metric=Euclidean())

    cluster_size_threshold >= 1 || throw(ArgumentError("cluster_size_threshold must be ≥ 1"))

    cls = clusters(butina_result)
    sort!(cls, by=length, rev=true)
    large_clusters = filter(c -> length(c) ≥ cluster_size_threshold, cls)
    isempty(large_clusters) && return butina_result

    new_assignments = similar(butina_result.assignments)
    new_counts = zeros(Int, length(large_clusters))

    for (cid, cluster) in enumerate(large_clusters)
        new_assignments[cluster] .= cid
        new_counts[cid] = length(cluster)
    end

    for sc in cls
        length(sc) ≥ cluster_size_threshold && continue

        for idx in sc
            best_cluster = argmin(point_to_cluster_distance(data, idx, lc, metric) for lc in large_clusters)
            new_assignments[idx] = best_cluster
            new_counts[best_cluster] += 1
        end

    end

    return ButinaResult(new_assignments, new_counts, butina_result.cutoff)
end

# HELPER FUNCTIONS


"""
    neighbour_lists(fps::AbstractVector,
                    threshold::Real,
                    metric::PreMetric)
        → NamedTuple{(:lists, :map),
                     Tuple{Vector{Vector{Int}}, Vector{Int}}}

* `lists[p]` holds the neighbour positions of the element at position `p`
  in `map`.

* `map[p]` gives the original index of `fps` that position `p` corresponds to.

The computation of distances is performed lazily - pair by pair - to reduce memory usage.


Return, for every element of `fps`, the indices of all other elements whose
distance ≤ `threshold` under `metric`.

"""
function neighbour_lists(fps::AbstractVector; threshold::Real, metric::PreMetric)

    map = collect(eachindex(fps))
    n = numobs(map)

    lists = [Int[] for _ in 1:n]

    for (i_pos, j_pos) in combinations(1:n, 2)
        i_idx, j_idx = map[i_pos], map[j_pos]

        Distances.evaluate(metric, fps[i_idx], fps[j_idx]) ≤ threshold || continue

        push!(lists[i_pos], j_pos)
        push!(lists[j_pos], i_pos)
    end

    return (lists=lists, map=map)
end
