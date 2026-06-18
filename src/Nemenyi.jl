export NemenyiResult, nemenyi, nemenyi_after_iman

using HypothesisTests
using Statistics, Distributions

"""
    NemenyiResult

Fields:
- avg_ranks::Vector{Float64}   # average rank per algorithm (lower is better if you ranked that way)
- se::Float64                   # standard error for rank differences: sqrt(k*(k+1)/(6n))
- cd::Float64                   # critical difference = qcrit * se
- pairs::Vector{NamedTuple}     # per-pair results: (i, j, diff, q, significant)
"""
struct NemenyiResult
    avg_ranks::Vector{Float64}
    se::Float64
    cd::Float64
    pairs::Vector{NamedTuple}
end


"""
    nemenyi(ft::FriedmanTest; alpha=0.05, qcrit=nothing)

Nemenyi post-hoc after a significant Friedman test.

Arguments:
- `ft`: a `FriedmanTest` object (from your implementation).
- `alpha`: family-wise α (default 0.05).
- `qcrit`: the studentized-range critical value q_{α; k, ∞}. If `nothing`,
           uses a conservative Bonferroni fallback.

Returns: `NemenyiResult`.

Notes:
- For exact Nemenyi, set `qcrit = qtukey(1 - alpha, k, Inf)` from a stats table or external lib.
  (In R: `stats::qtukey(1 - alpha, k, Inf)`; in SciPy: `scipy.stats.tukeylambda` equivalents or `scipy.stats.studentized_range` if available.)
"""
function nemenyi(ft::FriedmanTest; alpha::Real=0.05, qcrit_sr::Union{Nothing,Real}=nothing)
    n, k = ft.n, ft.k
    R = ft.rank_sums ./ n
    se = sqrt(k * (k + 1) / (6n))

    qdist = StudentizedRange(Inf, k)
    q_sr = qcrit_sr === nothing ? quantile(qdist, 1 - alpha) : float(qcrit_sr)
    cd = (q_sr / sqrt(2)) * se  # Demšar/PMCMR CD

    pairs = NamedTuple[]
    for i in 1:k-1, j in i+1:k
        diff = R[i] - R[j]
        q_obs = abs(diff) / se * sqrt(2)        # studentized-range scale
        # Guard: NaN/Inf q_obs (from degenerate data) → treat as non-significant
        p = (isnan(q_obs) || isinf(q_obs)) ? 1.0 : 1 - cdf(qdist, q_obs)
        reject = p < alpha                    # equivalent to abs(diff) > cd
        push!(pairs, (i=i, j=j, diff=diff, q_obs=q_obs, p_value=p, reject=reject))
    end

    NemenyiResult(collect(float.(R)), se, cd, pairs)
end

"""
    nemenyi_after_iman(ft; alpha=0.05, qcrit=nothing, run_anyway=false)

Convenience wrapper:
1) Computes Iman–Davenport F and p.
2) If p < alpha (or `run_anyway=true`), runs Nemenyi and returns it; otherwise returns `nothing`.

Returns: (F, pF, nemenyi_result_or_nothing)
"""
function nemenyi_after_iman(ft::FriedmanTest; alpha::Real=0.05, qcrit::Union{Nothing,Real}=nothing, run_anyway::Bool=false)
    F, pF = iman_davenport(ft)
    if pF < alpha || run_anyway
        return F, pF, nemenyi(ft; alpha=alpha, qcrit=qcrit)
    else
        return F, pF, nothing
    end
end
