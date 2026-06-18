using CSV
using DataFrames
using Statistics
using StatsBase: tiedrank, corspearman, corkendall

export dataset_names,
    metrics_results,
    merge_repeated_run_results,
    mean_scores_by_set,
    benchmark_summary,
    benchmark_gap,
    invert_to_minimize,
    is_higher_better,
    HIGHER_BETTER_DEFAULT,
    perdataset_means,
    average_ranks_across_datasets,
    splitter_family,
    short_code,
    collapse_dataset_algorithm,
    per_dataset_spread_df,
    quadrant_summary_df,
    threshold_counts_df,
    find_homogeneous_rank_spans,
    layout_group_bar_rows,
    pairwise_to_matrix,
    bedroc,
    compute_fold_metrics,
    compute_ad_coverage,
    nemenyi_from_df,
    compute_fold_selection_quality,
    aggregate_selection_by_dataset

const HIGHER_BETTER_DEFAULT = Dict(
    :R2 => true,
    :RMSE => false,
    :MAE => false,
    :MedAE => false,
    :MAPE => false,
    :Pearson => true,
    :Spearman => true,
    :Kendall => true,
)

metric_symbol(metric::Union{Symbol,AbstractString}) = Symbol(metric)

function is_higher_better(metric::Union{Symbol,AbstractString};
    higher_better::AbstractDict{Symbol,Bool}=HIGHER_BETTER_DEFAULT)
    m = metric_symbol(metric)
    get(higher_better, m, nothing) !== nothing || throw(ArgumentError("unknown metric direction for $m"))
    return higher_better[m]
end

"""
    dataset_names(data_root::AbstractString) -> Vector{String}

Return dataset base names found under `data_root`, assuming files follow the
`<name>_training.csv` / `<name>_test.csv` convention.
"""
function dataset_names(data_root::AbstractString)
    names = String[]
    for file in readdir(data_root)
        endswith(file, ".csv") || continue
        m = match(r"^(.*?)(?:_training|_test)\.csv$", file)
        m === nothing || push!(names, m.captures[1])
    end
    return sort!(unique(names))
end

"""
    metrics_results(dataset, algorithm; base_dir) -> DataFrame

Load `metrics.csv` for a specific `dataset`/`algorithm` combination. Returns an
empty `DataFrame` if the file is missing.
"""
function metrics_results(
    dataset::AbstractString,
    algorithm::AbstractString;
    base_dir::AbstractString,
    expected_rows::Union{Nothing,Int}=nothing,
)
    path = joinpath(base_dir, dataset, algorithm, "metrics.csv")
    if !isfile(path)
        @warn "metrics.csv not found" dataset = dataset algorithm = algorithm path = path
        return DataFrame()
    end

    df = CSV.read(path, DataFrame)
    if expected_rows !== nothing && nrow(df) != expected_rows
        @warn "Incorrect number of metrics rows" dataset = dataset algorithm = algorithm size = nrow(df) expected = expected_rows
    end
    return df
end

"""
    metrics_results(dataset; algorithms, base_dir) -> DataFrame

Load and concatenate metric files for all `algorithms`, annotating the
`algorithm` column. Missing files are skipped.
"""
function metrics_results(
    dataset::AbstractString;
    algorithms::AbstractVector{<:AbstractString},
    base_dir::AbstractString,
    expected_rows::Union{Nothing,Int}=nothing,
)
    dfs = DataFrame[]
    for algorithm in algorithms
        df = metrics_results(dataset, algorithm; base_dir=base_dir, expected_rows=expected_rows)
        if !isempty(df)
            df[!, :algorithm] = fill(algorithm, nrow(df))
            push!(dfs, df)
        end
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

const _DEFAULT_MERGE_METRICS = [:R2, :RMSE, :MAE, :MedAE, :MAPE, :Pearson, :Spearman, :Kendall]

"""
    merge_repeated_run_results(df; metrics) -> DataFrame

Average repeated CV runs grouped by `:dataset`, `:algorithm`, `:rep`, `:fold`
and `:set`. Pass `metrics` to override the default metric columns.
"""
function merge_repeated_run_results(
    dataset::DataFrame;
    metrics::AbstractVector{<:Union{Symbol,String}}=_DEFAULT_MERGE_METRICS,
)
    combine(
        groupby(dataset, [:dataset, :algorithm, :rep, :fold, :set]),
        Symbol.(metrics) .=> mean,
        renamecols=false,
    )
end

"""
    mean_scores_by_set(df; metrics, label_col=:set)

Return per-(algorithm, dataset, metric, `label_col`) mean scores.
"""
function mean_scores_by_set(
    df::DataFrame;
    metrics::AbstractVector{<:Union{Symbol,String}},
    label_col::Symbol=:set,
)
    metric_syms = Symbol.(metrics)
    long = stack(df, metric_syms; variable_name=:metric, value_name=:score)
    combine(
        groupby(long, [:algorithm, :dataset, :metric, label_col]),
        :score => mean => :score,
    )
end

"""
    benchmark_gap(internal, external, metric; higher_better=HIGHER_BETTER_DEFAULT)

Sign convention from the manuscript (Δ > 0 = optimistic, i.e. internal overestimates):
- higher-is-better metrics: Δ = internal - external
- lower-is-better metrics:  Δ = external - internal
"""
benchmark_gap(internal, external, metric::Union{String,Symbol}; higher_better=HIGHER_BETTER_DEFAULT) = is_higher_better(metric; higher_better=higher_better) ? (internal - external) : (external - internal)

"""
    benchmark_summary(df; metric, metrics, label_col=:set, higher_better=HIGHER_BETTER_DEFAULT)

Return a `DataFrame` with columns `:algorithm`, `:dataset`, `:ips`, `:test`,
and `:diff` (benchmark gap) for the selected metric.
"""
function benchmark_summary(
    df::DataFrame;
    metric::Union{String,Symbol},
    metrics::AbstractVector{<:Union{Symbol,String}},
    label_col::Symbol=:set,
    higher_better=HIGHER_BETTER_DEFAULT,
)
    metric_sym = Symbol(metric)
    per_set = mean_scores_by_set(df; metrics=metrics, label_col=label_col)
    filtered = filter(:metric => ==(metric_sym), per_set)
    combine(groupby(filtered, [:algorithm, :dataset])) do s
        labels = s[!, label_col]
        ips_vals = skipmissing(s.score[labels.=="ips"])
        test_vals = skipmissing(s.score[labels.=="test"])
        ips = isempty(ips_vals) ? missing : mean(ips_vals)
        test = isempty(test_vals) ? missing : mean(test_vals)
        (; ips, test, diff=(ismissing(ips) || ismissing(test)) ? missing : benchmark_gap(ips, test, metric_sym; higher_better=higher_better))
    end
end

"""
    invert_to_minimize(x, metric; higher_better=HIGHER_BETTER_DEFAULT)

Flip the sign of `x` when the metric is higher-better, useful when converting to
rankings where lower is preferred.
"""
invert_to_minimize(x, metric::Union{String,Symbol}; higher_better=HIGHER_BETTER_DEFAULT) = is_higher_better(metric; higher_better=higher_better) ? -x : x

# ---------------------------------------------------------------------------
# Rank-based analysis helpers
# ---------------------------------------------------------------------------

"""
    perdataset_means(df, metric; set="ips", metrics) -> DataFrame

Return per-(dataset, algorithm) mean scores for `metric` and a given `set` label.
"""
function perdataset_means(
    df::DataFrame,
    metric::Symbol;
    set::AbstractString="ips",
    metrics::AbstractVector{<:Union{Symbol,String}}=collect(keys(HIGHER_BETTER_DEFAULT)),
)
    metric_syms = Symbol.(metrics)
    long = stack(df, metric_syms; variable_name=:metric, value_name=:score)
    sub = filter(row -> row.set == set && Symbol(row.metric) == metric, long)
    isempty(sub) && return DataFrame()
    combine(groupby(sub, [:dataset, :algorithm]), :score => mean => :score)
end

"""
    average_ranks_across_datasets(scores, metric; higher_better=HIGHER_BETTER_DEFAULT) -> DataFrame

Rank algorithms per dataset, then average ranks. Lower rank = better.
"""
function average_ranks_across_datasets(
    scores::DataFrame,
    metric::Union{Symbol,String};
    higher_better::AbstractDict{Symbol,Bool}=HIGHER_BETTER_DEFAULT,
)
    hb = is_higher_better(metric; higher_better=higher_better)
    parts = DataFrame[]
    for g in groupby(scores, :dataset)
        s = sort(DataFrame(g), :score; rev=hb)
        push!(parts, DataFrame(dataset=s.dataset, algorithm=s.algorithm, rank=1:nrow(s)))
    end
    ranks = vcat(parts...)
    avg = combine(groupby(ranks, :algorithm), :rank => mean => :avg_rank, nrow => :n)
    avg[!, :Ndatasets] .= length(unique(scores.dataset))
    sort!(avg, :avg_rank)
    return avg
end

# ---------------------------------------------------------------------------
# Splitter metadata helpers
# ---------------------------------------------------------------------------

"""
    splitter_family(alg) -> String

Map an algorithm name to a high-level family label.
"""
function splitter_family(alg::AbstractString)
    alg in ("kennardstone", "mdks", "morais", "spxy-euclidean", "spxy-jaccard") && return "Kennard–Stone"
    alg in ("optisim", "minimum_dissimilarity", "maximum_dissimilarity") && return "Diversity"
    alg == "random" && return "Random"
    occursin("stratified", alg) && return "Clustering (Stratified)"
    (alg == "butina" || occursin("shuffle", alg)) && return "Clustering (Shuffle)"
    return "Clustering (Other)"
end

"""
    short_code(alg) -> String

Return a short display code for an algorithm name.
"""
function short_code(alg::AbstractString)
    alg == "kennardstone"          && return "KS"
    alg == "mdks"                  && return "MDKS"
    alg == "morais"                && return "Morais"
    alg == "spxy-euclidean"        && return "SPXY"
    alg == "spxy-jaccard"          && return "SPXY-J"
    alg == "optisim"               && return "OptiSim"
    alg == "minimum_dissimilarity" && return "MinDis"
    alg == "maximum_dissimilarity" && return "MaxDis"
    alg == "random"                && return "Rand"
    alg == "butina"                && return "Butina"
    alg == "kmeans_stratified"     && return "KM-strat"
    alg == "kmeans_shuffle"        && return "KM-shuf"
    alg == "kmedoids_stratified"   && return "KMed-strat"
    alg == "kmedoids_shuffle"      && return "KMed-shuf"
    alg == "hac_stratified"        && return "HAC-strat"
    alg == "hac_shuffle"           && return "HAC-shuf"

    s = replace(alg, "_" => "-")
    has_umap = occursin("umap", s)
    base = has_umap ? "U" : ""
    occursin("butina", s)       && (base *= "B")
    occursin("hierarchical", s) && (base *= "H")
    occursin("kmeans", s)       && (base *= "KM")
    occursin("kmedoids", s)     && (base *= "KMed")
    isempty(base) || base == "U" && (base *= "C")
    st = occursin("stratified", s) ? "strat" : "shuf"
    return "$(base)-$(st)"
end

# ---------------------------------------------------------------------------
# Summary helpers for visualisation data
# ---------------------------------------------------------------------------

"""
    collapse_dataset_algorithm(df) -> DataFrame

Average `ips` and `diff` per (dataset, algorithm).
"""
function collapse_dataset_algorithm(df::DataFrame)
    combine(groupby(df, [:dataset, :algorithm])) do s
        _ips  = filter(isfinite, collect(skipmissing(s.ips)))
        _diff = filter(isfinite, collect(skipmissing(s.diff)))
        ips  = isempty(_ips)  ? NaN : mean(_ips)
        diff = isempty(_diff) ? NaN : mean(_diff)
        (; ips, diff)
    end
end

"""
    per_dataset_spread_df(df) -> DataFrame

Return min/max/range of benchmark value and estimation error per dataset.
"""
function per_dataset_spread_df(df::DataFrame)
    combine(groupby(df, :dataset)) do s
        bench = filter(isfinite, collect(skipmissing(s.ips)))
        err   = filter(isfinite, collect(skipmissing(s.diff)))
        isempty(bench) && (bench = [0.0])
        isempty(err)   && (err   = [0.0])
        (;
            bench_min   = minimum(bench),
            bench_max   = maximum(bench),
            bench_range = maximum(bench) - minimum(bench),
            err_min     = minimum(err),
            err_max     = maximum(err),
            err_range   = maximum(err) - minimum(err),
        )
    end
end

"""
    quadrant_summary_df(df) -> DataFrame

Return per-algorithm median benchmark value and median estimation bias.
"""
function quadrant_summary_df(df::DataFrame)
    out = combine(groupby(df, :algorithm)) do s
        bench = filter(isfinite, collect(skipmissing(s.ips)))
        err   = filter(isfinite, collect(skipmissing(s.diff)))
        bench_med = isempty(bench) ? NaN : median(bench)
        err_med   = isempty(err)   ? NaN : median(err)
        (; bench_med, err_med)
    end
    out.label = String.(out.algorithm)
    return out
end

"""
    threshold_counts_df(df, metric; thr, directional, higher_better) -> DataFrame

Count, per algorithm, how many datasets have |Δ| > `thr`, split by direction.
"""
function threshold_counts_df(
    df::DataFrame,
    metric::Union{String,Symbol};
    thr::Real=0.15,
    directional::Bool=true,
    higher_better::AbstractDict{Symbol,Bool}=HIGHER_BETTER_DEFAULT,
)
    hb = is_higher_better(metric; higher_better=higher_better)
    optimistic_mask(d) = hb ? (d < -thr) : (d >  thr)
    pessimistic_mask(d) = hb ? (d >  thr) : (d < -thr)

    out = combine(groupby(df, :algorithm)) do s
        diffs  = collect(skipmissing(s.diff))
        abs_ct = count(d -> abs(d) > thr, diffs)
        opt_ct  = directional ? count(optimistic_mask,  diffs) : missing
        pess_ct = directional ? count(pessimistic_mask, diffs) : missing
        (; abs_ct, opt_ct, pess_ct, Ndatasets=length(unique(s.dataset)))
    end
    out.label = String.(out.algorithm)
    sort!(out, :abs_ct; rev=true)
    return out
end

# ---------------------------------------------------------------------------
# CD diagram geometry helpers
# ---------------------------------------------------------------------------

"""
    find_homogeneous_rank_spans(ranks, cd) -> Vector{Tuple{Float64,Float64}}

Return maximal non-significant homogeneous groups as (lo, hi) rank spans.
"""
function find_homogeneous_rank_spans(ranks::AbstractVector{<:Real}, cd::Real; tol=1e-12)
    k = length(ranks)
    spans = Tuple{Float64,Float64}[]
    for i in 1:k-1
        j = i + 1
        while j <= k && (ranks[j] - ranks[i] <= cd + tol)
            j += 1
        end
        j - i >= 2 && push!(spans, (float(ranks[i]), float(ranks[j-1])))
    end
    spans = sort(spans; by=s -> -(s[2] - s[1]))
    keep = Tuple{Float64,Float64}[]
    for s in spans
        any(t -> s[1] >= t[1] - tol && s[2] <= t[2] + tol, keep) || push!(keep, s)
    end
    return keep
end

"""
    layout_group_bar_rows(spans; y_base, row_step) -> (placed, y_max)

Assign non-overlapping vertical rows to homogeneous-group bars.
"""
function layout_group_bar_rows(
    spans::AbstractVector{<:Tuple};
    y_base::Real=1.0,
    row_step::Real=0.25,
    tol::Real=1e-12,
)
    placed = Tuple{Float64,Float64,Float64}[]
    for s in spans
        y = float(y_base)
        while any(p -> (max(p[1], s[1]) <= min(p[2], s[2]) + tol) && abs(p[3] - y) < 1e-9, placed)
            y += row_step
        end
        push!(placed, (float(s[1]), float(s[2]), y))
    end
    y_max = isempty(placed) ? float(y_base) : maximum(p[3] for p in placed)
    return placed, y_max
end

# ---------------------------------------------------------------------------
# Pairwise result helpers
# ---------------------------------------------------------------------------

"""
    pairwise_to_matrix(df; value, diag, mode, order) -> (Matrix, labels)

Convert a pairwise DataFrame (columns :group1, :group2, `value`) to a square matrix.
`mode` = :antisymmetric (default), :symmetric, or :copy.
"""
function pairwise_to_matrix(
    df::DataFrame;
    value::Symbol=:mean_diff,
    diag::Real=0.0,
    mode::Symbol=:antisymmetric,
    order=nothing,
)
    g1   = String.(df.group1)
    g2   = String.(df.group2)
    vals = Float64.(df[!, value])

    labels = if order === nothing
        sort!(unique!(vcat(g1, g2)))
    else
        ord  = String.(collect(order))
        extra = setdiff(sort!(unique!(vcat(g1, g2))), ord)
        vcat(ord, extra)
    end

    n   = length(labels)
    pos = Dict(l => i for (i, l) in enumerate(labels))
    M   = fill(NaN, n, n)
    for i in 1:n; M[i, i] = float(diag); end

    for k in eachindex(vals)
        i = pos[g1[k]]; j = pos[g2[k]]; v = vals[k]
        M[j, i] = v
        mode === :antisymmetric && (M[i, j] = -v)
        mode === :symmetric     && (M[i, j] =  v)
    end
    return M, labels
end

# ---------------------------------------------------------------------------
# Friedman–Nemenyi pipeline helper
# ---------------------------------------------------------------------------

"""
    nemenyi_from_df(df, row_col, col_col, val_col;
                    higher_better=false, absolute=false, alpha=0.05)
        -> (NemenyiResult, Vector{String})

Run the complete Friedman–Nemenyi pipeline on a long-format DataFrame:
unstack to wide, drop rows with any missing, build a Float64 matrix,
optionally take `abs` (for gap/error metrics) or negate (for higher-is-better
metrics so that rank 1 goes to the best), then call `FriedmanTest` and
`nemenyi`.

Returns the `NemenyiResult` and the ordered column labels (the `col_col`
values that survived `dropmissing`), so callers do not need a separate unstack
just to recover the label order.
"""
function nemenyi_from_df(
    df::DataFrame,
    row_col::Symbol,
    col_col::Symbol,
    val_col::Symbol;
    higher_better::Bool=false,
    absolute::Bool=false,
    alpha::Real=0.05,
)
    wide   = DataFrames.unstack(df, row_col, col_col, val_col) |> dropmissing
    labels = names(wide)[2:end]
    # Guard: need ≥2 complete rows for a valid Friedman test
    if nrow(wide) < 2 || length(labels) < 2
        @warn "nemenyi_from_df: insufficient complete data ($(nrow(wide)) rows, $(length(labels)) cols) — returning NaN result"
        k = length(labels)
        return NemenyiResult(fill(NaN, k), NaN, NaN, NamedTuple[]), labels
    end
    M      = Matrix{Float64}(wide[:, 2:end])
    absolute     && (M = abs.(M))
    higher_better && (M = -M)
    ft = FriedmanTest(eachcol(M)...)
    return nemenyi(ft; alpha=alpha), labels
end

# ---------------------------------------------------------------------------
# Per-fold prediction metrics
# ---------------------------------------------------------------------------

"""
    top_k_enrichment(y_true, y_pred; k=0.10) -> Float64

Enrichment factor (EF) at the top-k% of the predicted ranking. Actives are the
true top-k% by value; the predicted top-k% is the selection. EF is the hit rate
in the selection divided by the hit rate expected at random:

    EF = (hits / n_selected) / (n_actives / n)

with `n_selected = n_actives = round(k·n)`. EF = 1 for random ranking, and
EF = 1/k for perfect ranking (its theoretical maximum). EF > 1 indicates
enrichment of true actives near the top of the predicted list.
"""
function top_k_enrichment(y_true::AbstractVector, y_pred::AbstractVector; k::Float64=0.10)
    n     = length(y_true)
    n_top = max(1, round(Int, k * n))
    true_top = Set(partialsortperm(y_true, 1:n_top; rev=true))
    pred_top = partialsortperm(y_pred, 1:n_top; rev=true)
    hits = count(i -> i in true_top, pred_top)
    # Enrichment factor = (hits/n_selected) / (n_actives/n); n_selected = n_actives = n_top.
    (hits / n_top) / (n_top / n)
end

"""
    bedroc(y_true, y_pred; α=20.0, ra=0.2) -> Float64

BEDROC (Boltzmann-Enhanced Discrimination of ROC; Truchon & Bayly, J. Chem. Inf. Model.
2007, 47, 488–508). Actives are defined as the top-`ra` fraction by true value.
α controls early-recovery sensitivity: α=20 concentrates ~80% of the weight on the
top 8% of the ranked list.

Returns a value in [0, 1]: ~`ra` for random ranking, 1 for perfect ranking.
"""
function bedroc(y_true::AbstractVector, y_pred::AbstractVector; α::Float64=20.0, ra::Float64=0.2)
    n   = length(y_true)
    n_a = max(1, round(Int, ra * n))
    Ra  = n_a / n

    true_top  = Set(partialsortperm(y_true, 1:n_a; rev=true))
    pred_order = sortperm(y_pred; rev=true)

    # 1-indexed ranks of actives in the predicted ranking
    ranks = [i for (i, idx) in enumerate(pred_order) if idx ∈ true_top]

    Se  = sum(exp(-α * r / n) for r in ranks)
    # Z_α = (1/N) Σ_{i=1}^{N} exp(−α i/N) — closed form
    Z_α = exp(-α / n) * (1 - exp(-α)) / (n * (1 - exp(-α / n)))
    RIE = Se / (Ra * n * Z_α)

    # Scale RIE to BEDROC ∈ [0, 1] (Truchon & Bayly, Eq. 15)
    RIE * Ra * sinh(α / 2) / (cosh(α / 2) - cosh(α / 2 - α * Ra)) +
        1 / (1 - exp(α * (1 - Ra)))
end

"""
    compute_fold_metrics(df; group_cols) -> DataFrame

Compute RMSE, MAE, MedAE, R², Pearson, Spearman, Kendall τ and top-5/10/20%
enrichment per group from a DataFrame with `y_true` and `y_pred` columns.
"""
function compute_fold_metrics(
    df::DataFrame;
    group_cols::AbstractVector{Symbol}=[:dataset, :splitter, :rep, :fold, :set],
)
    combine(groupby(df, group_cols)) do g
        y  = Float64.(g.y_true)
        ŷ  = Float64.(g.y_pred)
        n  = length(y)
        if n < 2
            return (; n, rmse=NaN, mae=NaN, medae=NaN, r2=NaN, pearson=NaN, spearman=NaN,
                      kendall=NaN, enrichment_5=NaN, enrichment_10=NaN, enrichment_20=NaN,
                      bedroc=NaN)
        end
        ss_res = sum((y .- ŷ).^2)
        ss_tot = sum((y .- mean(y)).^2)
        resid  = abs.(y .- ŷ)
        (;
            n,
            rmse          = sqrt(mean((y .- ŷ).^2)),
            mae           = mean(resid),
            medae         = median(resid),
            r2            = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN,
            pearson       = cor(y, ŷ),
            spearman      = corspearman(y, ŷ),
            kendall       = corkendall(y, ŷ),
            enrichment_5  = top_k_enrichment(y, ŷ; k=0.05),
            enrichment_10 = top_k_enrichment(y, ŷ; k=0.10),
            enrichment_20 = top_k_enrichment(y, ŷ; k=0.20),
            bedroc        = bedroc(y, ŷ),
        )
    end
end

"""
    compute_fold_selection_quality(bench_metrics) -> DataFrame

From a DataFrame with per-(dataset, splitter, rep, fold, model) **external benchmark** metrics
(set == "ips") plus `cv_rms` and `is_champion` columns, compute per-fold selection quality:

- `champ_adv_rmse/pearson/spearman`: champion's external benchmark score minus the mean of
  all alternatives' external scores (positive = champion generalises better on average)
- `best_adv_rmse/pearson/spearman`: champion's external score minus the best alternative's
  external score (oracle regret; positive = champion is also the externally best model)
- `hit_rmse/pearson/spearman`: 1 if the champion is the externally best model, 0 otherwise
"""
function compute_fold_selection_quality(df::DataFrame)
    valid = filter(df) do r
        !ismissing(r.cv_rms) && !isnan(r.rmse) && !isnan(r.pearson) && !isnan(r.spearman)
    end
    disallowmissing!(valid, :cv_rms)
    combine(groupby(valid, [:dataset, :splitter, :rep, :fold])) do g
        n = nrow(g)
        champ_idx = findfirst(x -> coalesce(x, false), g.is_champion)
        (isnothing(champ_idx) || n < 2) && return (;
            champ_adv_rmse=missing, champ_adv_pearson=missing, champ_adv_spearman=missing,
            best_adv_rmse=missing, best_adv_pearson=missing, best_adv_spearman=missing,
            hit_rmse=missing, hit_pearson=missing, hit_spearman=missing,
        )
        disc = setdiff(1:n, [champ_idx])
        (;
            champ_adv_rmse     = mean(g.rmse[disc])    - g.rmse[champ_idx],
            champ_adv_pearson  = g.pearson[champ_idx]  - mean(g.pearson[disc]),
            champ_adv_spearman = g.spearman[champ_idx] - mean(g.spearman[disc]),
            best_adv_rmse      = minimum(g.rmse[disc]) - g.rmse[champ_idx],
            best_adv_pearson   = g.pearson[champ_idx]  - maximum(g.pearson[disc]),
            best_adv_spearman  = g.spearman[champ_idx] - maximum(g.spearman[disc]),
            hit_rmse           = Float64(g.rmse[champ_idx]      <= minimum(g.rmse)),
            hit_pearson        = Float64(g.pearson[champ_idx]   >= maximum(g.pearson)),
            hit_spearman       = Float64(g.spearman[champ_idx]  >= maximum(g.spearman)),
        )
    end
end

"""
    aggregate_selection_by_dataset(fold_sel) -> (sel_by_ds, hitrate_by_ds)

Average fold-level champion selection quality per (dataset, splitter) for use
in Friedman–Nemenyi tests. Returns two DataFrames:
- `sel_by_ds`: per-(dataset, splitter) mean `adv_rmse/pearson/spearman` and `best_adv_*`
- `hitrate_by_ds`: per-(dataset, splitter) fraction of folds where champion is externally best
"""
function aggregate_selection_by_dataset(fold_sel::DataFrame)
    sel_by_ds = combine(
        groupby(dropmissing(fold_sel, [:champ_adv_rmse, :best_adv_rmse]), [:dataset, :splitter]),
        :champ_adv_rmse     => mean => :adv_rmse,
        :champ_adv_pearson  => mean => :adv_pearson,
        :champ_adv_spearman => mean => :adv_spearman,
        :best_adv_rmse      => mean => :best_adv_rmse,
        :best_adv_pearson   => mean => :best_adv_pearson,
        :best_adv_spearman  => mean => :best_adv_spearman,
    )
    _clean(col) = [v for v in col if !ismissing(v) && !isnan(Float64(v))]
    hitrate_by_ds = combine(groupby(fold_sel, [:dataset, :splitter])) do g
        rv = _clean(g.hit_rmse)
        pv = _clean(g.hit_pearson)
        sv = _clean(g.hit_spearman)
        (;
            hitrate_rmse     = isempty(rv) ? missing : mean(rv),
            hitrate_pearson  = isempty(pv) ? missing : mean(pv),
            hitrate_spearman = isempty(sv) ? missing : mean(sv),
        )
    end
    return sel_by_ds, hitrate_by_ds
end

"""
    compute_ad_coverage(df; set, group_cols) -> DataFrame

Compute the fraction of predictions in `set` that fall within the applicability
domain (`in_domain == true`) per group.
"""
function compute_ad_coverage(
    df::DataFrame;
    set::AbstractString="ips",
    group_cols::AbstractVector{Symbol}=[:dataset, :splitter, :rep, :fold],
)
    sub = filter(:set => ==(set), df)
    combine(groupby(sub, group_cols)) do g
        n    = nrow(g)
        n_in = count(coalesce(x, false) for x in g.in_domain)
        (; n_samples=n, n_in_domain=n_in, coverage=n_in / n)
    end
end
