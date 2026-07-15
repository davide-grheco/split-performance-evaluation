# Multicriteria decision analysis (MCDA) over the eleven evaluation measures.
#
# Answers Reviewer 1's request for a formal MCDA (TOPSIS / MOORA) that distils an overall
# ranking of the splitting strategies. To respect the paper's own point that raw metric
# values (RMSE, MAE, …) are not comparable across datasets, every MCDA is run WITHIN each
# dataset (15 splitters × criteria) and the resulting per-dataset ranks are then averaged
# across the fifteen datasets — the same rank-aggregation used everywhere else in the paper.
#
# Three views (the paper's two objectives, plus their combination):
#   * performance  — 11 criteria = external-benchmark performance in each metric.
#   * credibility  — 11 criteria = |estimation gap| in each metric (cost: smaller is better).
#   * combined     — all 22 criteria together, equal weight (⇒ the two objectives are
#                    balanced 50/50 since each contributes 11 equally-weighted criteria).
#
# Reuses exactly the Figure-1 data (`all_metrics_avgs`: per splitter × dataset × metric, with
# external-benchmark performance `ips` and signed estimation gap `diff`). The heavy load step
# (AD prediction files) is cached to results/mcda_all_metrics_avgs.arrow for fast re-runs.
#
# Run:  julia --project=. scripts/generate_mcda.jl
# Outputs: results/mcda_rankings.csv, images/<EXPERIMENT>/mcda_tradeoff.{pdf,png}

using Arrow, DataFrames, Statistics, StatsBase, CairoMakie, CategoricalArrays

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "Friedman.jl"))
include(joinpath(SRC, "Nemenyi.jl"))
include(joinpath(SRC, "MCDA.jl"))
include(joinpath(SRC, "AnalysisUtils.jl"))
include(joinpath(SRC, "Plotting.jl"))
include(joinpath(SRC, "ExperimentLoader.jl"))

# ------------------------------------------------------------------
# Constants (kept in sync with scripts/generate_article_figures.jl)
# ------------------------------------------------------------------
const EXPERIMENT      = get(ENV, "EXPERIMENT", "revision_v1")
const EXPERIMENT_ROOT = joinpath(@__DIR__, "..", "experiments", EXPERIMENT)
const OUTDIR = let d = get(ENV, "FIG_DIR", joinpath(@__DIR__, "..", "images", EXPERIMENT))
    isdir(d) || mkpath(d); d
end
const RESULTS_DIR = let d = joinpath(@__DIR__, "..", "results"); isdir(d) || mkpath(d); d end
const CACHE = joinpath(RESULTS_DIR, "mcda_all_metrics_avgs.arrow")

const DATASETS = [
    "3A4", "CB1", "DPP4", "HIVINT", "HIVPROT", "LOGD",
    "METAB", "NK1", "OX1", "OX2", "PGP", "PPB", "RAT_F", "TDI", "THROMBIN",
]
const SPLITTERS = [
    "random", "kennardstone", "mdks", "butina",
    "spxy-jaccard", "optisim", "maximum_dissimilarity",
    "minimum_dissimilarity", "morais",
    "kmeans_stratified", "kmeans_shuffle",
    "kmedoids_stratified", "kmedoids_shuffle",
    "hac_stratified", "hac_shuffle",
]
const SPLITTER_HUMAN_MAP = Dict(
    "random"=>"Random", "kennardstone"=>"Kennard–Stone", "mdks"=>"MDKS",
    "butina"=>"Butina (Shuffle)", "spxy-jaccard"=>"SPXY", "optisim"=>"OptiSim",
    "maximum_dissimilarity"=>"Maximum Dissimilarity", "minimum_dissimilarity"=>"Minimum Dissimilarity",
    "morais"=>"Morais", "kmeans_stratified"=>"K-Means (Stratified)", "kmeans_shuffle"=>"K-Means (Shuffle)",
    "kmedoids_stratified"=>"K-Medoids (Stratified)", "kmedoids_shuffle"=>"K-Medoids (Shuffle)",
    "hac_stratified"=>"HAC (Stratified)", "hac_shuffle"=>"HAC (Shuffle)",
)
const METRICS = [:RMSE, :R2, :MAE, :MedAE, :Pearson, :Spearman, :Kendall,
                 :Enrichment5, :Enrichment10, :Enrichment20, :BEDROC]
const HIGHER_BETTER = Dict(
    :RMSE=>false, :R2=>true, :MAE=>false, :MedAE=>false, :Pearson=>true,
    :Spearman=>true, :Kendall=>true, :Enrichment5=>true, :Enrichment10=>true,
    :Enrichment20=>true, :BEDROC=>true,
)

# Family colour/marker coding (Okabe–Ito), matching the other article figures.
const FAMILY_COLOR = Dict(
    "Kennard–Stone"=>Makie.to_color("#D55E00"), "Clustering (Shuffle)"=>Makie.to_color("#0072B2"),
    "Clustering (Stratified)"=>Makie.to_color("#56B4E9"), "Diversity"=>Makie.to_color("#009E73"),
    "Random"=>Makie.to_color("#000000"),
)
const FAMILY_MARKER = Dict(
    "Kennard–Stone"=>:diamond, "Clustering (Shuffle)"=>:circle,
    "Clustering (Stratified)"=>:utriangle, "Diversity"=>:rect, "Random"=>:star5,
)
const FAMILY_ORDER = ["Kennard–Stone", "Clustering (Shuffle)",
                      "Clustering (Stratified)", "Diversity", "Random"]

# ------------------------------------------------------------------
# Build (or load cached) all_metrics_avgs: per (algorithm, dataset, metric),
# external-benchmark performance `ips` and signed estimation gap `diff`.
# ------------------------------------------------------------------
function build_all_metrics_avgs()
    if isfile(CACHE)
        @info "Loading cached all_metrics_avgs from $CACHE"
        return DataFrame(Arrow.Table(CACHE))
    end
    @info "Loading layer1 summaries..."
    _raw_main  = load_layer1_summary(EXPERIMENT_ROOT; datasets=DATASETS)
    _raw_clust = load_layer1_summary(EXPERIMENT_ROOT; datasets=DATASETS,
        layer1_subdir=joinpath("layer1_summary", "new_splitters"))
    all_long_base = layer1_to_long(vcat(_raw_main, _raw_clust; cols=:union))

    @info "Loading AD summary (reads prediction files — a few minutes)..."
    ad_summary = load_ad_summary(EXPERIMENT_ROOT; datasets=DATASETS)

    @info "Merging repeated runs..."
    all_results = let _join_cols = [:dataset, :algorithm, :rep, :fold, :set],
                      _joined    = leftjoin(all_long_base, ad_summary.fold_pearson_spearman; on=_join_cols)
        merge_repeated_run_results(_joined; metrics=METRICS)
    end

    @info "Aggregating to (algorithm, dataset, metric)..."
    _stacked = stack(all_results, METRICS; variable_name=:metric, value_name=:score)
    _per_set = combine(groupby(_stacked, [:algorithm, :dataset, :metric, :set]),
        :score => mean => :score)
    avgs = combine(groupby(_per_set, [:algorithm, :dataset, :metric])) do s
        lbl  = s.set
        ips  = let v = skipmissing(s.score[lbl .== "ips"]);  isempty(v) ? missing : mean(v); end
        test = let v = skipmissing(s.score[lbl .== "test"]); isempty(v) ? missing : mean(v); end
        m    = Symbol(s.metric[1])
        diff = (ismissing(ips) || ismissing(test)) ? missing :
               benchmark_gap(test, ips, m; higher_better=HIGHER_BETTER)
        (; ips, test, diff)
    end
    Arrow.write(CACHE, avgs)
    @info "Cached all_metrics_avgs → $CACHE"
    return avgs
end

# ------------------------------------------------------------------
# MCDA: within each dataset build a decision matrix (splitters × criteria),
# run TOPSIS + MOORA, then average the per-dataset ranks across datasets.
# `criterion_value(row, metric)` extracts the criterion cell; `benefit_of(metric)`
# its direction. Returns a tidy DataFrame with mean_rank and mean_score.
# ------------------------------------------------------------------
"""
    run_view(avgs, view) -> DataFrame

`view` ∈ (:performance, :credibility, :combined). Aggregates within-dataset TOPSIS/MOORA
ranks over datasets. Columns: splitter, view, method, mean_rank, mean_score, n_datasets.
"""
function run_view(avgs::DataFrame, view::Symbol; metric_weights::Union{Nothing,Dict}=nothing)
    # (criterion label, extractor column, benefit?) list for this view.
    perf_crit = [(m, :ips, HIGHER_BETTER[m]) for m in METRICS]
    cred_crit = [(m, :diff, false) for m in METRICS]  # value is |diff|, cost
    crit = view === :performance ? perf_crit :
           view === :credibility ? cred_crit :
           view === :combined    ? vcat(perf_crit, cred_crit) :
           error("unknown view $view")

    # Equal per-metric weight by default; `metric_weights` (metric ⇒ weight) enables the
    # per-construct robustness check. Weights are renormalised inside topsis/moora, so only
    # relative values matter.
    weights = metric_weights === nothing ? ones(length(crit)) :
              Float64[metric_weights[c[1]] for c in crit]
    benefit = Bool[c[3] for c in crit]

    # Accumulate ranks/scores per splitter across datasets.
    topsis_ranks  = Dict(s => Float64[] for s in SPLITTERS)
    moora_ranks   = Dict(s => Float64[] for s in SPLITTERS)
    topsis_scores = Dict(s => Float64[] for s in SPLITTERS)
    moora_scores  = Dict(s => Float64[] for s in SPLITTERS)

    for ds in DATASETS
        sub = filter(:dataset => ==(ds), avgs)
        # Build the decision matrix: rows = splitters present with complete criteria.
        rows = String[]
        mat  = Vector{Float64}[]
        for s in SPLITTERS
            ssub = filter(:algorithm => ==(s), sub)
            vals = Float64[]
            ok = true
            for (m, col, _) in crit
                r = filter(:metric => ==(String(m)), ssub)
                if nrow(r) != 1 || ismissing(r[1, col])
                    ok = false; break
                end
                v = Float64(r[1, col])
                col === :diff && (v = abs(v))   # credibility criterion = |gap|
                push!(vals, v)
            end
            ok || continue
            push!(rows, s); push!(mat, vals)
        end
        length(rows) >= 2 || (@warn "dataset $ds: <2 complete splitters, skipped"; continue)
        X = permutedims(hcat(mat...))          # (n_splitters × n_criteria)

        rt = topsis_rank(X, weights, benefit)
        rm = moora_rank(X, weights, benefit)
        for (i, s) in enumerate(rows)
            push!(topsis_ranks[s],  rt.ranks[i]);  push!(topsis_scores[s], rt.scores[i])
            push!(moora_ranks[s],   rm.ranks[i]);  push!(moora_scores[s],  rm.scores[i])
        end
    end

    out = DataFrame(splitter=String[], view=Symbol[], method=String[],
                    mean_rank=Float64[], mean_score=Float64[], n_datasets=Int[])
    for s in SPLITTERS
        isempty(topsis_ranks[s]) && continue
        push!(out, (s, view, "TOPSIS", mean(topsis_ranks[s]), mean(topsis_scores[s]), length(topsis_ranks[s])))
        push!(out, (s, view, "MOORA",  mean(moora_ranks[s]),  mean(moora_scores[s]),  length(moora_ranks[s])))
    end
    out
end

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
avgs = build_all_metrics_avgs()

results = vcat(run_view(avgs, :performance),
               run_view(avgs, :credibility),
               run_view(avgs, :combined))
results.splitter_label = [get(SPLITTER_HUMAN_MAP, s, s) for s in results.splitter]
results.family = splitter_family.(results.splitter)

# Guard the "mean ranks out of 15" guarantee: every splitter must be ranked in every
# dataset, otherwise per-splitter means would average ranks over different (and differently
# sized) dataset subsets. Fails loudly rather than biasing silently on incomplete inputs.
let bad = filter(r -> r.n_datasets != length(DATASETS), results)
    isempty(bad) || error("Incomplete dataset coverage (expected $(length(DATASETS))): " *
        join(unique(["$(r.splitter)/$(r.view)=$(r.n_datasets)" for r in eachrow(bad)]), ", "))
end

CSV_OUT = joinpath(RESULTS_DIR, "mcda_rankings.csv")
using CSV
CSV.write(CSV_OUT, results)
@info "Wrote $CSV_OUT ($(nrow(results)) rows)"

# Console summary: combined-view TOPSIS ranking (best first).
@info "=== Combined-view TOPSIS overall ranking (lower mean rank = better) ==="
_comb = sort(filter(r -> r.view === :combined && r.method == "TOPSIS", results), :mean_rank)
show(select(_comb, :splitter_label, :family, :mean_rank, :mean_score); allrows=true, allcols=true)
println()

# ------------------------------------------------------------------
# Weight robustness (addresses the correlated-measures concern): several measures track
# closely, so equal per-measure weight lets a correlated block act as repeated votes. Recompute
# the combined ranking with one weight per construct — error {RMSE,MAE,MedAE}, explained
# variance {R²}, correlation {Pearson,Spearman,Kendall}, enrichment {EF5/10/20,BEDROC} — each
# construct contributing equally, and report the rank correlation against the equal-per-measure
# ranking. Applied to both the performance and credibility blocks, so the 50/50 balance holds.
# ------------------------------------------------------------------
const CONSTRUCT = Dict(
    :RMSE=>:error, :MAE=>:error, :MedAE=>:error, :R2=>:variance,
    :Pearson=>:corr, :Spearman=>:corr, :Kendall=>:corr,
    :Enrichment5=>:enrich, :Enrichment10=>:enrich, :Enrichment20=>:enrich, :BEDROC=>:enrich,
)
let
    counts = countmap(collect(values(CONSTRUCT)))          # members per construct
    wmap   = Dict(m => 1.0 / counts[CONSTRUCT[m]] for m in METRICS)  # each construct sums to 1
    alt    = run_view(avgs, :combined; metric_weights=wmap)
    a = sort(filter(r -> r.method == "TOPSIS", alt), :splitter)
    b = sort(filter(r -> r.view === :combined && r.method == "TOPSIS", results), :splitter)
    rho   = corspearman(a.mean_rank, b.mean_rank)
    top3(df)  = Set(sort(df, :mean_rank).splitter[1:3])
    bot2(df)  = Set(sort(df, :mean_rank; rev=true).splitter[1:2])
    @info "Weight robustness (combined TOPSIS): equal-per-measure vs equal-per-construct " *
          "Spearman ρ = $(round(rho, digits=3)); top-3 identical: $(top3(a)==top3(b)); " *
          "bottom-2 identical: $(bot2(a)==bot2(b))"
end

# ------------------------------------------------------------------
# Trade-off figure: performance-MCDA rank vs credibility-MCDA rank (TOPSIS), one point per
# splitter. Points near the anti-diagonal indicate no splitter is best on both objectives.
# ------------------------------------------------------------------
function tradeoff_df(method::AbstractString)
    p = filter(r -> r.view === :performance && r.method == method, results)
    c = filter(r -> r.view === :credibility && r.method == method, results)
    df = innerjoin(select(p, :splitter, :splitter_label, :family, :mean_rank => :perf_rank),
                   select(c, :splitter, :mean_rank => :cred_rank); on=:splitter)
    df.code = short_code.(df.splitter)
    df
end

let df = tradeoff_df("TOPSIS")
    fig = Figure(size=(820, 720), fontsize=15)
    ax = Axis(fig[1, 1];
        title="Multicriteria trade-off (TOPSIS)",
        subtitle="Each axis aggregates the eleven measures into one rank",
        xlabel="Benchmark-performance rank (1 = best)",
        ylabel="Estimation-credibility rank (1 = best)")
    n = nrow(df)
    lines!(ax, [1, n], [n, 1]; color=(:gray, 0.4), linestyle=:dash)  # perfect anti-diagonal
    fams = [f for f in FAMILY_ORDER if f in df.family]
    for f in fams
        sub = filter(:family => ==(f), df)
        scatter!(ax, sub.perf_rank, sub.cred_rank;
            color=(FAMILY_COLOR[f], 0.85), marker=FAMILY_MARKER[f],
            markersize=15, strokewidth=0.8, strokecolor=:black, label=f)
    end
    for (i, r) in enumerate(eachrow(df))
        # Alternate label placement to reduce overlap of near-coincident points.
        al = isodd(i) ? (:left, :bottom) : (:left, :top)
        text!(ax, r.perf_rank, r.cred_rank; text="  " * r.code,
            fontsize=10, align=al)
    end
    Legend(fig[1, 2], ax, "Family"; framevisible=false)
    save(joinpath(OUTDIR, "mcda_tradeoff.pdf"), fig)
    save(joinpath(OUTDIR, "mcda_tradeoff.png"), fig; px_per_unit=2)
    @info "Wrote mcda_tradeoff.{pdf,png}"
end

@info "MCDA analysis complete."
