using Random
using Distances
using Dates
using UMAP
using Clustering
using DataSplits
using MLUtils


# Random
# Stratified Random
# Hierarchical Clustering
# UMAP Clustering
# SIMPD
# MaxMin
# Kennard-Stone
# Locality-sensitive Hashing
# DUPLEX algorithm

export butina_split, umap_split, time_split, random_split, make_cluster_splitter, apply_split, optisim

check_fraction(frac::Real) = 0 < frac < 1 || throw(ArgumentError("frac must be between 0 and 1"))

"""
    butina_split(fingerprints; cutoff=0.2, frac=0.8, rng=Random.GLOBAL_RNG)

Perform a Butina‐clustering‐based split:

- `fingerprints` `Vector{BitVector}` of precomputed fingerprints, or
- `cutoff` is the Tanimoto‐distance threshold (default 0.2).
- `frac` is the fraction of compounds to assign to the **train** set.
- `rng` is the random number generator for shuffling clusters.

# Returns
A NamedTuple `(train, test)` where
- `train::Vector{Int}` are the training indices,
- `test::Vector{Int}` are the testing indices.
"""
function butina_split(fingerprints::AbstractMatrix{T};
    cutoff::Real=0.2,
    frac::Real=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG) where {T}

    r = butina(fingerprints; threshold=cutoff, metric=Jaccard())
    return cluster_split(r; frac=frac, rng=rng)
end

function butina_split(fingerprints::AbstractVector{BitVector};
    cutoff::Real=0.2,
    frac::Real=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    r = butina(fingerprints; threshold=cutoff, metric=Jaccard())
    return cluster_split(r; frac=frac, rng=rng)
end


function cluster_split(clusters::ClusteringResult;
    frac::Real=0.8,
    rng::AbstractRNG=Random.GLOBAL_RNG)

    check_fraction(frac)

    assignments = clusters.assignments
    total_points = numobs(assignments)
    unique_clusters = unique(assignments)
    shuffled_clusters = copy(unique_clusters)
    shuffle!(rng, shuffled_clusters)

    train = Int[]
    for cluster_id in shuffled_clusters
        if numobs(train) / total_points < frac
            cluster_points = findall(==(cluster_id), assignments)
            append!(train, cluster_points)
        else
            break
        end
    end

    train = unique(train)
    test = setdiff(1:total_points, train)

    return (train=train, test=test)
end


function time_split(dates::Vector{<:Date}; frac::Real=0.8)
    check_fraction(frac)

    idx = sortperm(dates)
    sorted_dates = dates[idx]

    split_at = floor(Int, frac * numobs(dates))
    cutoff_date = sorted_dates[split_at]

    train = findall(<(cutoff_date), dates)
    test = findall(>=(cutoff_date), dates)

    return (train=train, test=test)
end

function random_split(n::Integer; frac::Real=0.8, rng::AbstractRNG=Random.GLOBAL_RNG)
    check_fraction(frac)
    n > 0 || throw(ArgumentError("n must be positive"))

    idx = randperm(rng, n)
    split_at = floor(Int, frac * n)
    (train=idx[1:split_at], test=idx[split_at+1:end])
end

random_split(data; kwargs...) = random_split(numobs(data); kwargs...)


function umap_split(fingerprints::AbstractMatrix{T};
    n_clusters::Int=10,
    frac::Real=0.8,
    n_neighbors::Int=15,
    metric::SemiMetric=Jaccard(),
    rng::AbstractRNG=Random.GLOBAL_RNG) where {T}

    check_fraction(frac)
    n = numobs(fingerprints)
    n ≥ 2 || throw(ArgumentError("need at least 2 samples for UMAP split"))

    nn = min(n_neighbors, n - 1)
    k = min(n_clusters, n)
    nn == n_neighbors || @warn "Adjusted n_neighbors to $nn for dataset size $n."
    k == n_clusters || @warn "Adjusted n_clusters to $k for dataset size $n."

    Z = UMAP.umap(Float64.(fingerprints); metric=metric, n_neighbors=nn)
    result = kmeans(Z, k; rng=rng)

    return cluster_split(result; frac=frac, rng=rng)
end



"""
    repeated_cv_splits(n, k, repeats; seed=nothing)

Return a `Vector` of `repeats * k` splits for repeated k-fold CV.

Each element is a `NamedTuple`:
`(train::Vector{Int}, test::Vector{Int}, repeat::Int, fold::Int)`

- `n`       : number of observations (indices are `1:n`)
- `k`       : number of folds per repeat (e.g., 5)
- `repeats` : how many times to repeat k-fold with a new shuffle
- `seed`    : optional RNG seed for reproducibility

Notes:
- Within each repeat, every observation appears in exactly one `test` fold.
- Fold sizes are balanced (differ by at most 1).
"""
function repeated_cv_splits(n::Int, k::Int, repeats::Int; seed::Union{Nothing,Int}=nothing)
    @assert n ≥ 2 "n must be ≥ 2"
    @assert 2 ≤ k ≤ n "k must be between 2 and n"
    @assert repeats ≥ 1 "repeats must be ≥ 1"

    rng = seed === nothing ? Random.default_rng() : MersenneTwister(seed)

    function _fold_slices(n::Int, k::Int)
        base, r = divrem(n, k)
        slices = Vector{UnitRange{Int}}(undef, k)
        start = 1
        for i in 1:k
            len = i ≤ r ? base + 1 : base
            stop = start + len - 1
            slices[i] = start:stop
            start = stop + 1
        end
        return slices
    end
    slices = _fold_slices(n, k)

    T = NamedTuple{(:train, :test, :repeat, :fold),
        Tuple{Vector{Int},Vector{Int},Int,Int}}
    out = Vector{T}()
    sizehint!(out, repeats * k)

    for r in 1:repeats
        perm = randperm(rng, n)
        @inbounds for j in 1:k
            test = perm[slices[j]]
            train = Vector{Int}(undef, n - length(test))
            pos = 1
            for t in 1:k
                t == j && continue
                s = slices[t]
                len = length(s)
                train[pos:pos+len-1] = perm[s]
                pos += len
            end
            push!(out, (train=train, test=test, repeat=r, fold=j))
        end
    end
    return out
end


"""
    make_cluster_splitter(clustering, name; frac=0.8) -> DataSplits.SplitStrategy

Build a cluster-aware split strategy from a pre-computed `ClusteringResult`.
`name` is `"stratified"` or `"shuffle"`.
"""
function make_cluster_splitter(clustering, name::AbstractString; frac::Real=0.8)
    name == "stratified" && return DataSplits.ClusterStratifiedSplit(clustering, :proportional; frac=frac)
    name == "shuffle" && return DataSplits.ClusterShuffleSplit(clustering, frac)
    error("Unknown cluster splitter \"$name\". Expected \"stratified\" or \"shuffle\".")
end


"""
    optisim(D, selected_samples, max_subsample_size, distance_cutoff; rng) -> Set{Int}

Optimized OptiSim kernel operating on a precomputed distance matrix `D`.
Maintains an incremental `min_dist` vector so each dissimilarity check is O(1)
rather than O(|selected|). Pass `max_subsample_size=0` for full-scan (maximum
dissimilarity) behaviour.
"""
function optisim(
    D::AbstractMatrix,
    selected_samples::Int = 10,
    max_subsample_size::Int = 0,
    distance_cutoff::Float64 = 0.35;
    rng = Random.default_rng(),
)
    N = size(D, 1)
    M = min(selected_samples, N)
    K = max_subsample_size

    candidates = collect(1:N)
    min_dist = fill(Inf, N)

    idx = rand(rng, 1:N)
    first_sel = candidates[idx]
    candidates[idx] = candidates[end]; pop!(candidates)
    selected = [first_sel]
    sizehint!(selected, M)

    @inbounds for i in candidates
        min_dist[i] = D[i, first_sel]
    end

    subsample = Int[]
    sizehint!(subsample, K == 0 ? N : K)
    shuffle_buf = Vector{Int}(undef, N)

    while length(selected) < M
        empty!(subsample)

        if K == 0 || K >= length(candidates)
            i = 1
            while i <= length(candidates)
                c = candidates[i]
                if min_dist[c] >= distance_cutoff
                    push!(subsample, c)
                    i += 1
                else
                    candidates[i] = candidates[end]; pop!(candidates)
                end
            end
        else
            nc = length(candidates)
            resize!(shuffle_buf, nc)
            @inbounds for i in 1:nc; shuffle_buf[i] = i; end
            n_drawn = 0
            j = 1
            while n_drawn < K && j <= length(shuffle_buf)
                ri = rand(rng, j:length(shuffle_buf))
                shuffle_buf[j], shuffle_buf[ri] = shuffle_buf[ri], shuffle_buf[j]
                c = candidates[shuffle_buf[j]]
                if min_dist[c] >= distance_cutoff
                    push!(subsample, c)
                    n_drawn += 1
                end
                j += 1
            end
        end

        isempty(subsample) && break

        best = subsample[1]; best_score = min_dist[best]
        @inbounds for i in subsample
            s = min_dist[i]
            if s > best_score; best_score = s; best = i; end
        end

        push!(selected, best)
        pos = findfirst(==(best), candidates)
        if pos !== nothing
            candidates[pos] = candidates[end]; pop!(candidates)
        end

        @inbounds for i in candidates
            d = D[i, best]
            if d < min_dist[i]; min_dist[i] = d; end
        end
    end

    return Set(selected)
end


"""
    apply_split(X, y, splitter; rng) -> NamedTuple(train, test)

Apply a `DataSplits.SplitStrategy`. Uses `hasmethod` to determine whether the
strategy's `split` implementation requires labels (`y`) or only features (`X`).
No exception handling — dispatch is resolved via Julia's method table.
"""
function apply_split(X, y, splitter::DataSplits.SplitStrategy;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    try
        return DataSplits.split(X, y, splitter; rng=rng)
    catch e
        e isa MethodError || rethrow()
        return DataSplits.split(X, splitter; rng=rng)
    end
end
