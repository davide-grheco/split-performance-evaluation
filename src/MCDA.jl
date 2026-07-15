# Multicriteria decision analysis (MCDA): thin, well-typed wrappers over JMcDM.
#
# We delegate the actual TOPSIS and MOORA computations to JMcDM.jl (a mature, tested
# package) rather than re-implementing them, and expose a single small interface tailored
# to this study:
#
#   topsis_rank(X, weights, benefit) -> MCDAResult
#   moora_rank(X, weights, benefit)  -> MCDAResult
#   mcda_rank(X, weights, benefit)   -> (; topsis, moora)
#
# where
#   X       : m×n decision matrix (rows = alternatives, columns = criteria),
#   weights : length-n criterion weights (need not be normalized; we normalize to sum 1),
#   benefit : length-n Bool vector, true when a larger value is better (benefit criterion),
#             false for cost criteria (smaller is better).
#
# Each result carries the JMcDM score (higher = better) and tie-aware ranks (1 = best,
# average ranks for ties) derived from those scores. Self-contained (`using JMcDM`,
# `StatsBase`) so it can be `include`d standalone by the analysis scripts, mirroring
# Friedman.jl / Nemenyi.jl.

export MCDAResult, topsis_rank, moora_rank, mcda_rank

using StatsBase: tiedrank
import JMcDM

"""
    MCDAResult

Fields:
- `scores::Vector{Float64}` — one aggregate score per alternative; higher is better.
- `ranks::Vector{Float64}`  — ranks derived from `scores` (1 = best); average ranks for ties.
"""
struct MCDAResult
    scores::Vector{Float64}
    ranks::Vector{Float64}
end

# JMcDM encodes criterion direction as a vector of functions: `maximum` for benefit,
# `minimum` for cost.
_direction_fns(benefit::AbstractVector{Bool}) = Function[b ? maximum : minimum for b in benefit]

# Coerce inputs to the concrete types JMcDM expects and validate shapes / weights.
function _prepare(X::AbstractMatrix, weights::AbstractVector, benefit::AbstractVector{Bool})
    m, n = size(X)
    length(weights) == n || throw(DimensionMismatch("weights length $(length(weights)) ≠ n criteria $n"))
    length(benefit) == n || throw(DimensionMismatch("benefit length $(length(benefit)) ≠ n criteria $n"))
    s = sum(weights)
    s > 0 || throw(ArgumentError("weights must sum to a positive value"))
    Xf = Matrix{Float64}(X)
    wf = Float64.(weights) ./ s
    return Xf, wf, _direction_fns(benefit)
end

"""
    topsis_rank(X, weights, benefit) -> MCDAResult

TOPSIS via `JMcDM.topsis`. The score is the closeness coefficient
C = D⁻ / (D⁺ + D⁻) ∈ [0, 1]; higher is better.
"""
function topsis_rank(X::AbstractMatrix, weights::AbstractVector, benefit::AbstractVector{Bool})
    Xf, wf, fns = _prepare(X, weights, benefit)
    res = JMcDM.topsis(Xf, wf, fns)
    scores = Float64.(res.scores)
    MCDAResult(scores, tiedrank(-scores))
end

"""
    moora_rank(X, weights, benefit) -> MCDAResult

MOORA (ratio system) via `JMcDM.moora`. The score is the overall assessment (Σ benefit −
Σ cost of the weighted, normalized criteria); higher is better.
"""
function moora_rank(X::AbstractMatrix, weights::AbstractVector, benefit::AbstractVector{Bool})
    Xf, wf, fns = _prepare(X, weights, benefit)
    # :ratio = the canonical MOORA ratio system (Σ benefit − Σ cost, higher = better);
    # JMcDM defaults to the reference-point (minimax) variant, which we do not want here.
    res = JMcDM.moora(Xf, wf, fns; method = :ratio)
    scores = Float64.(res.scores)
    MCDAResult(scores, tiedrank(-scores))
end

"""
    mcda_rank(X, weights, benefit) -> NamedTuple

Run both methods on the same decision matrix. Returns `(; topsis, moora)`, each an
`MCDAResult`.
"""
mcda_rank(X::AbstractMatrix, weights::AbstractVector, benefit::AbstractVector{Bool}) =
    (; topsis = topsis_rank(X, weights, benefit), moora = moora_rank(X, weights, benefit))
