export nn_tanimoto
using Distances
using HypothesisTests
using LinearAlgebra, Statistics
using DataSplits
using MLUtils
using Distributions

include("Metrics/MMD.jl")

export tanimoto, evaluate_split_metrics, mahalanobis_split_distance, nn_distance, nn_distance_batch, tukey_hsd,
    AbstractSplitMetrics, SplitMetrics, FingerprintSplitMetrics

abstract type AbstractSplitMetrics end

struct SplitMetrics <: AbstractSplitMetrics
    train_samples::Int
    test_samples::Int
    num_features::Int
    train_mean::Float64
    test_mean::Float64
    sparsity_gap::Float64
    mahalanobis_distance::Float64
end

struct FingerprintSplitMetrics <: AbstractSplitMetrics
    train_samples::Int
    test_samples::Int
    num_features::Int
    train_mean::Float64
    test_mean::Float64
    sparsity_gap::Float64
    mahalanobis_distance::Float64
    nn_tanimoto_mean::Float64
    nn_tanimoto_std::Float64
    js_divergence::Float64
    coverage_score::Float64
end

function Base.Dict(m::SplitMetrics)
    Dict(
        "Train_Samples" => m.train_samples,
        "Test_Samples" => m.test_samples,
        "Num_Features" => m.num_features,
        "Train_Mean" => m.train_mean,
        "Test_Mean" => m.test_mean,
        "Sparsity_Gap" => m.sparsity_gap,
        "mahalanobis_split_distance" => m.mahalanobis_distance,
    )
end

function Base.Dict(m::FingerprintSplitMetrics)
    Dict(
        "Train_Samples" => m.train_samples,
        "Test_Samples" => m.test_samples,
        "Num_Features" => m.num_features,
        "Train_Mean" => m.train_mean,
        "Test_Mean" => m.test_mean,
        "Sparsity_Gap" => m.sparsity_gap,
        "mahalanobis_split_distance" => m.mahalanobis_distance,
        "NN_Tanimoto_Mean" => m.nn_tanimoto_mean,
        "NN_Tanimoto_Std" => m.nn_tanimoto_std,
        "JS_Divergence" => m.js_divergence,
        "Coverage_Score" => m.coverage_score,
    )
end

"""
    tanimoto(a, b) -> Float64
Bit-level Tanimoto similarity of two fingerprints.
"""
@inline function tanimoto(a::BitVector, b::BitVector)
    length(a) == length(b) || throw(DimensionMismatch())
    return 1 - jaccard(a, b)
end

"""
    nn_distance(train, test; metric=Euclidean()) -> Vector{Float64}

For every observation in `test` return the *minimum* distance to any
observation in `train`, according to `metric`.

Arguments
---------
- `train`, `test`  : either `AbstractMatrix` (rows = samples, cols = features)
                     **or** `Vector{<:AbstractVector}` (list of fingerprints).
- `metric`         : anything accepted by *Distances.jl* (`Euclidean()`,
                     `CosineDist()`, `Jaccard()`, …) **or** a plain function
                     `metric(x, y)` returning a real scalar.

Returns
-------
A `Vector{Float64}` of length `numobs(test)` with the nearest-neighbour
distances.
"""
function nn_distance(train::AbstractMatrix, test::AbstractMatrix; metric=Distances.Euclidean())
    # train, test: [features, samples]
    D = pairwise(metric, test, train; dims=2)
    return vec(map(minimum, eachrow(D)))
end

function nn_distance(train::AbstractVector{<:AbstractVector}, test::AbstractVector{<:AbstractVector}; metric::SemiMetric=Distances.Euclidean())
    train = reduce(hcat, train)
    test = reduce(hcat, test)
    return nn_distance(train, test, metric=metric)
end

function nn_distance_batch(train, test; metric::Distances.SemiMetric=Distances.Euclidean(), stride=4096)

    stride ≤ 0 && error("`stride` must be a positive integer or `nothing`")

    tr = train isa AbstractMatrix ? train : reduce(hcat, train)
    te = test isa AbstractMatrix ? test : reduce(hcat, test)


    Nts, Ntr = numobs(te), numobs(tr)
    dmin = fill(typemax(Float64), Nts)

    for first in 1:stride:Nts
        last = min(first + stride - 1, Nts)
        block = getobs(te, first:last)
        Dblk = Distances.pairwise(metric, block, tr; dims=2)
        dmin[first:last] .= min.(dmin[first:last],
            map(minimum, eachrow(Dblk)))
    end
    return dmin
end

"""
    nn_tanimoto(train, test) -> Vector{Float64}

For each fingerprint in `test`, compute its maximum Tanimoto
similarity against **any** fingerprint in `train`.
"""
function nn_tanimoto(train::Vector{BitVector},
    test::Vector{BitVector})::Vector{Float64}
    return [maximum(tanimoto(t, tr) for tr in eachobs(train)) for t in eachobs(test)]
end

"""
    bit_frequency_js(train, test) -> Float64

Compute the Jensen–Shannon divergence between the per-bit occurrence
frequencies of two binary fingerprint matrices.

Each row is a fingerprint bit; the frequency vector is the mean
across columns (samples). Returns a scalar divergence value in [0, log(2)].
"""
function bit_frequency_js(train::BitMatrix, test::BitMatrix)
    p = mapslices(mean, train; dims=2) |> vec       # freq ∈ [0,1]
    q = mapslices(mean, test; dims=2) |> vec
    js = jensen_shannon(p, q)
    return js
end

function sparsity_gap(train::BitMatrix, test::BitMatrix)::Float64
    return abs(mean(train) - mean(test))
end

"Coverage score (bits present in test also present somewhere in train)."
function coverage_score(train::BitMatrix, test::BitMatrix)::Float64
    covered = any(train; dims=2)                  # bool vector length B
    needed = any(test; dims=2)
    return sum(covered .& needed) / sum(needed)   # ∈ [0,1]
end


"""
    mahalanobis_split_distance(train::AbstractMatrix, test::AbstractMatrix) -> Float64

Computes the symmetric Mahalanobis-based distributional distance (Λ)
between training and test sets, as proposed by Jain et al. (2022). This
quantifies whether a given train/test split is statistically representative
of the overall dataset.

Assumes both datasets are continuous and have the same number of columns.

# Arguments
- `train::AbstractMatrix`: Matrix of training samples (rows = samples, cols = features).
- `test::AbstractMatrix`: Matrix of test samples.

# Returns
- `Float64`: Symmetric Mahalanobis distance Λ.
"""
function mahalanobis_split_distance(train::AbstractMatrix, test::AbstractMatrix)::Float64
    n_train, n_test = numobs(train), numobs(test)

    μ_train = mean(train, dims=2) |> vec
    μ_test  = mean(test,  dims=2) |> vec

    Σ_pooled = pooled_covariance([train, test])

    train_c = Float64.(train) .- μ_test
    test_c  = Float64.(test)  .- μ_train

    # Try Cholesky on the unmodified covariance first so results are identical to
    # prior runs on full-rank matrices. When n < p the matrix is rank-deficient;
    # fall back to a ridge-regularised Cholesky instead of pinv, whose divide-and-
    # conquer SVD (gesdd) can fail to converge on highly degenerate inputs.
    F = cholesky(Symmetric(Σ_pooled), check=false)
    if !issuccess(F)
        p = size(Σ_pooled, 1)
        ε = 1e-6 * tr(Σ_pooled) / p
        F = cholesky(Symmetric(Σ_pooled + ε * I))
    end

    Ltr = F.L \ train_c
    Lte = F.L \ test_c

    return 0.5 * (sum(abs2, Ltr) / n_train + sum(abs2, Lte) / n_test)
end


"""
    evaluate_split_metrics(train::AbstractMatrix, test::AbstractMatrix) -> SplitMetrics

Generic version that computes basic dataset statistics like sparsity and dimensionality.
Can be specialized for specific matrix types.
"""
function evaluate_split_metrics(train::AbstractMatrix, test::AbstractMatrix)
    return SplitMetrics(
        numobs(train),
        numobs(test),
        length(getobs(train, 1)),
        mean(train),
        mean(test),
        abs(mean(train) - mean(test)),
        mahalanobis_split_distance(train, test),
    )
end

"""
    evaluate_split_metrics(train::BitMatrix, test::BitMatrix) -> FingerprintSplitMetrics

Specialized version for binary fingerprint matrices. Computes:
- NN Tanimoto similarity (mean, std)
- Jensen–Shannon divergence of bit frequencies
- Sparsity gap
- Coverage score
"""
function evaluate_split_metrics(train::BitMatrix, test::BitMatrix)
    base = evaluate_split_metrics(Matrix{Float64}(train), Matrix{Float64}(test))

    train_fps = collect(eachobs(train))
    test_fps = collect(eachobs(test))

    nn_sims = nn_tanimoto(train_fps, test_fps)

    return FingerprintSplitMetrics(
        base.train_samples,
        base.test_samples,
        base.num_features,
        base.train_mean,
        base.test_mean,
        base.sparsity_gap,
        base.mahalanobis_distance,
        mean(nn_sims),
        std(nn_sims),
        bit_frequency_js(train, test),
        coverage_score(train, test),
    )
end

"""
    pooled_covariance(groups::Vector{<:AbstractMatrix}) -> Matrix{Float64}

Compute the pooled covariance matrix from a list of data matrices (`groups`), assuming all groups share the same true covariance.

Each matrix in `groups` is one group (rows = features, columns = samples).
"""
function pooled_covariance(groups::Vector{<:AbstractMatrix})::Matrix{Float64}
    total_weight = 0
    p = size(groups[1], 1)
    pooled_cov = zeros(Float64, p, p)

    for group in groups
        n = numobs(group)
        n <= 1 && continue
        μ = mean(group, dims=2)
        X_c = group .- μ          # F×N; avoids allocating the N×F transpose
        pooled_cov .+= X_c * X_c' # accumulate unnormalised scatter
        total_weight += n - 1
    end

    total_weight > 0 || throw(ArgumentError("Total weight must be > 0"))
    return pooled_cov / total_weight
end

"""
    directed_hausdorff(train, test; metric=Euclidean()) -> Float64

One-sided (directed) Hausdorff distance

    h(test, train) = max_{x ∈ test}  min_{y ∈ train}  d(x, y)

* `train`, `test` can be **either**
  - `AbstractMatrix` (rows = samples, cols = features), **or**
  - `Vector{<:AbstractVector}` (e.g. `Vector{BitVector}` of fingerprints).
* `metric` accepts anything that works with `Distances.pairwise`, or a
  plain function `metric(x, y)` returning a real number.
"""
function directed_hausdorff(train, test; metric=Distances.Euclidean())
    return maximum(nn_distance(train, test; metric=metric))
end

"""
    hausdorff(A, B; metric = Euclidean()) -> Float64

Symmetric (undirected) Hausdorff distance

    H(A, B) = max( h(A, B),  h(B, A) )
"""
function hausdorff(A, B; metric=Distances.Euclidean())
    return max(directed_hausdorff(A, B; metric=metric), directed_hausdorff(B, A; metric=metric))
end




"""
    tukey_hsd(values, groups; alpha=0.05)

Pure Julia implementation of Tukey's HSD test using only Statistics and Distributions.

# Arguments
- `values`: Vector of measurements.
- `groups`: Vector of group labels (same length as values).
- `alpha`: Significance level (default 0.05).

# Returns
A vector of named tuples: (group1, group2, mean_diff, lower, upper, p_value, reject)
"""
function tukey_hsd(values, groups; alpha=0.05)
    group_levels = unique(groups)
    k = length(group_levels)
    idxs = [findall(==(g), groups) for g in group_levels]
    ns = [length(idxs[i]) for i in 1:k]
    means = [mean(values[idxs[i]]) for i in 1:k]
    grand_mean = mean(values)
    ssw = sum(sum((values[idxs[i]] .- means[i]) .^ 2) for i in 1:k)
    df_error = length(values) - k
    mse = ssw / df_error
    results = []
    qdist = StudentizedRange(df_error, k)
    for i = 1:k-1, j = i+1:k
        ni, nj = ns[i], ns[j]
        se = sqrt(mse / 2 * (1 / ni + 1 / nj))
        diff = means[i] - means[j]
        q_crit = quantile(qdist, 1 - alpha)
        ci = q_crit * se
        q_obs = abs(diff) / se * sqrt(2)
        pval = 1 - cdf(qdist, q_obs)
        reject = pval < alpha
        push!(results, (group1=group_levels[i], group2=group_levels[j], mean_diff=diff, lower=diff - ci, upper=diff + ci, p_value=pval, reject=reject))
    end
    return results
end
