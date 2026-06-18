using Distances
using LinearAlgebra
using Statistics
using MLUtils

export AbstractAD, BoundingBoxAD, KNNDistanceAD, LeverageAD, TanimotoAD
export fit_ad, ad_score, in_domain

abstract type AbstractAD end

# ─── Bounding Box ─────────────────────────────────────────────────────────────

"""
    BoundingBoxAD

Applicability domain based on per-feature min/max range of the training set.
A query point is in the AD if every feature lies within the training range.
"""
struct BoundingBoxAD <: AbstractAD
    feature_min::Vector{Float64}
    feature_max::Vector{Float64}
end

"""
    fit_ad(BoundingBoxAD, X) -> BoundingBoxAD

Fit a bounding-box AD to `X` (features × samples).
"""
function fit_ad(::Type{BoundingBoxAD}, X::AbstractMatrix)
    BoundingBoxAD(vec(minimum(X; dims=2)), vec(maximum(X; dims=2)))
end

"""
    ad_score(ad::BoundingBoxAD, X) -> Vector{Float64}

Fraction of features outside the training bounding box per query point.
0 means fully inside; 1 means all features are out of range.
"""
function ad_score(ad::BoundingBoxAD, X::AbstractMatrix)
    p = length(ad.feature_min)
    map(eachobs(X)) do x
        count(i -> x[i] < ad.feature_min[i] || x[i] > ad.feature_max[i], 1:p) / p
    end
end

in_domain(ad::BoundingBoxAD, X::AbstractMatrix) = iszero.(ad_score(ad, X))

# ─── k-NN Distance ────────────────────────────────────────────────────────────

"""
    KNNDistanceAD

AD based on the mean distance to the k nearest training neighbors.
A point is in the AD if its kNN mean distance does not exceed `threshold`.
"""
struct KNNDistanceAD <: AbstractAD
    train::Matrix{Float64}
    threshold::Float64
    k::Int
    metric::Distances.SemiMetric
end

"""
    fit_ad(KNNDistanceAD, X; k=5, z=1.5, metric=CosineDist()) -> KNNDistanceAD

Fit a kNN-distance AD to training data `X` (features × samples).

The threshold is auto-calibrated via leave-one-out:
`threshold = mean(loo_knn_mean) + z * std(loo_knn_mean)`.
"""
function fit_ad(::Type{KNNDistanceAD}, X::AbstractMatrix;
    k::Int=5, z::Float64=1.5,
    metric::Distances.SemiMetric=CosineDist())
    tr = Matrix{Float64}(X)
    n = numobs(tr)
    D = pairwise(metric, tr; dims=2)
    loo_means = map(1:n) do i
        row = copy(D[:, i])
        row[i] = Inf
        mean(partialsort(row, 1:min(k, n - 1)))
    end
    threshold = mean(loo_means) + z * std(loo_means)
    KNNDistanceAD(tr, threshold, k, metric)
end

"""
    ad_score(ad::KNNDistanceAD, X) -> Vector{Float64}

Mean distance to the k nearest training neighbors for each query point.
"""
function ad_score(ad::KNNDistanceAD, X::AbstractMatrix)
    D = pairwise(ad.metric, Matrix{Float64}(X), ad.train; dims=2)
    map(eachrow(D)) do dists
        mean(partialsort(dists, 1:min(ad.k, numobs(ad.train))))
    end
end

in_domain(ad::KNNDistanceAD, X::AbstractMatrix) = ad_score(ad, X) .<= ad.threshold

# ─── Leverage (Williams plot) ─────────────────────────────────────────────────

"""
    LeverageAD

Leverage-based AD (Williams plot). A query point is in the AD if its hat value
h_i = xᵢᵀ (XᵀX)⁻¹ xᵢ ≤ h* = 3(p+1)/n,
where p is the number of features and n is the number of training points.
An intercept term is included by default.
"""
struct LeverageAD <: AbstractAD
    XtXinv::Matrix{Float64}
    threshold::Float64
end

"""
    fit_ad(LeverageAD, X) -> LeverageAD

Fit leverage AD to training matrix `X` (features × samples).
Standard threshold h* = 3(p+1)/n.
"""
function fit_ad(::Type{LeverageAD}, X::AbstractMatrix)
    p = size(X, 1)
    n = numobs(X)
    Xb = vcat(Matrix{Float64}(X), ones(1, n))   # augment with intercept row
    XtXinv = pinv(Xb * Xb')
    threshold = 3(p + 1) / n
    LeverageAD(XtXinv, threshold)
end

"""
    ad_score(ad::LeverageAD, X) -> Vector{Float64}

Hat (leverage) value h_i = xᵢᵀ (XᵀX)⁻¹ xᵢ for each query point.
"""
function ad_score(ad::LeverageAD, X::AbstractMatrix)
    Xb = vcat(Matrix{Float64}(X), ones(1, numobs(X)))
    [dot(xi, ad.XtXinv * xi) for xi in eachcol(Xb)]
end

in_domain(ad::LeverageAD, X::AbstractMatrix) = ad_score(ad, X) .<= ad.threshold

# ─── Tanimoto (fingerprints) ─────────────────────────────────────────────────

"""
    TanimotoAD

Fingerprint-based AD: a query compound is in the AD if its maximum Tanimoto
similarity to any training compound is at least `threshold`.
"""
struct TanimotoAD <: AbstractAD
    train_fps::Vector{BitVector}
    threshold::Float64
end

"""
    fit_ad(TanimotoAD, fps; z=1.0) -> TanimotoAD

Calibrate a Tanimoto AD via leave-one-out on `fps`.
`threshold = clamp(mean(loo_sim) - z * std(loo_sim), 0, 1)`.
"""
function fit_ad(::Type{TanimotoAD}, fps::Vector{BitVector}; z::Float64=1.0)
    n = numobs(fps)
    loo_sims = map(1:n) do i
        train_loo = getobs(fps, filter(!=(i), 1:n))
        maximum(tanimoto(getobs(fps, i), fp) for fp in eachobs(train_loo))
    end
    threshold = clamp(mean(loo_sims) - z * std(loo_sims), 0.0, 1.0)
    TanimotoAD(fps, threshold)
end

"""
    ad_score(ad::TanimotoAD, fps) -> Vector{Float64}

Maximum Tanimoto similarity to any training fingerprint for each query.
Higher values indicate stronger support (more similar to training data).
"""
ad_score(ad::TanimotoAD, fps::Vector{BitVector}) = nn_tanimoto(ad.train_fps, fps)

in_domain(ad::TanimotoAD, fps::Vector{BitVector}) = ad_score(ad, fps) .>= ad.threshold
