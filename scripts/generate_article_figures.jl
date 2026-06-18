# Standalone figure generation script — no DataSplitBench import (avoids Chemistry.jl/CondaPkg).
# Saves all article figures to article/figs_v2/.
# Run with: julia --project=. scripts/generate_article_figures.jl

using Arrow, DataFrames, Statistics, StatsBase, CairoMakie, ColorSchemes,
      AlgebraOfGraphics, CategoricalArrays, PooledArrays, CSV

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "Friedman.jl"))
include(joinpath(SRC, "Nemenyi.jl"))
include(joinpath(SRC, "AnalysisUtils.jl"))
include(joinpath(SRC, "Plotting.jl"))
include(joinpath(SRC, "ExperimentLoader.jl"))

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
# Which experiment to render. Defaults to the full-data run (revision_v1), the
# canonical/primary results for the article. Override with EXPERIMENT=revision_v1_sub1000
# (and set FIG_DIR=images/revision_v1_sub1000) to regenerate the sensitivity figures.
# NOTE: the default output dir below is images/revision_v1 — keep EXPERIMENT and FIG_DIR
# consistent so a sub1000 run never overwrites the full-data figures.
const EXPERIMENT = get(ENV, "EXPERIMENT", "revision_v1")
const EXPERIMENT_ROOT = joinpath(@__DIR__, "..", "experiments", EXPERIMENT)

# Set SKIP_CHAMPION=1 to skip loading the (large) champion predictions. The
# champion figures (cd_adv, cd_rankcor, champ_freq) are then not regenerated.
# Use this to regenerate only the layer-1 / AD figures with much lower memory.
const SKIP_CHAMPION = haskey(ENV, "SKIP_CHAMPION")

# Output directory — defined early so static figures can be saved immediately
# after creation (avoids holding all Figure objects in memory simultaneously).
const OUTDIR = let d = get(ENV, "FIG_DIR", joinpath(@__DIR__, "..", "images", EXPERIMENT))
    isdir(d) || mkpath(d)
    d
end
_save_fig(stem, fig) = (isnothing(fig) || save(joinpath(OUTDIR, "$(stem).pdf"), fig); nothing)

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
    "random"               => "Random",
    "kennardstone"         => "Kennard–Stone",
    "mdks"                 => "MDKS",
    "butina"               => "Butina (Shuffle)",
    "spxy-jaccard"         => "SPXY",
    "optisim"              => "OptiSim",
    "maximum_dissimilarity"=> "Maximum Dissimilarity",
    "minimum_dissimilarity"=> "Minimum Dissimilarity",
    "morais"               => "Morais",
    "kmeans_stratified"    => "K-Means (Stratified)",
    "kmeans_shuffle"       => "K-Means (Shuffle)",
    "kmedoids_stratified"  => "K-Medoids (Stratified)",
    "kmedoids_shuffle"     => "K-Medoids (Shuffle)",
    "hac_stratified"       => "HAC (Stratified)",
    "hac_shuffle"          => "HAC (Shuffle)",
)

const METRICS = [:RMSE, :R2, :MAE, :MedAE, :Pearson, :Spearman, :Kendall,
                 :Enrichment5, :Enrichment10, :Enrichment20, :BEDROC]
const METRICS_STRS = String.(METRICS)
const METRIC_DISPLAY = Dict(
    "R2"          => "R²",
    "Enrichment5" => "EF-5%",
    "Enrichment10"=> "EF-10%",
    "Enrichment20"=> "EF-20%",
)
const HIGHER_BETTER = Dict(
    :RMSE         => false,
    :R2           => true,
    :MAE          => false,
    :MedAE        => false,
    :Pearson      => true,
    :Spearman     => true,
    :Kendall      => true,
    :Enrichment5  => true,
    :Enrichment10 => true,
    :Enrichment20 => true,
    :BEDROC       => true,
)

const PLOT_WIDTH  = 1500
const PLOT_HEIGHT = 600
const COLOR_SCHEME = :glasbey_bw_minc_20_maxl_70_n256

# ------------------------------------------------------------------
# Canonical family color/marker coding — applied identically across every
# family-coded figure (violins, scatters, bars, per-dataset spread) so that
# a given method family is shown the same way in all panels. Okabe–Ito
# colorblind-safe palette. Keys MUST match splitter_family() output exactly.
# ------------------------------------------------------------------
const FAMILY_COLOR = Dict(
    "Kennard–Stone"           => Makie.to_color("#D55E00"),  # vermillion
    "Clustering (Shuffle)"    => Makie.to_color("#0072B2"),  # blue
    "Clustering (Stratified)" => Makie.to_color("#56B4E9"),  # sky blue
    "Diversity"               => Makie.to_color("#009E73"),  # green
    "Random"                  => Makie.to_color("#000000"),  # black
)
const FAMILY_MARKER = Dict(
    "Kennard–Stone"           => :diamond,
    "Clustering (Shuffle)"    => :circle,
    "Clustering (Stratified)" => :utriangle,
    "Diversity"               => :rect,
    "Random"                  => :star5,
)
const FAMILY_ORDER = ["Kennard–Stone", "Clustering (Shuffle)",
                      "Clustering (Stratified)", "Diversity", "Random"]

"AoG scales mapping family → canonical color and marker."
family_cm_scales() = scales(
    Color  = (; palette = [f => FAMILY_COLOR[f]  for f in FAMILY_ORDER]),
    Marker = (; palette = [f => FAMILY_MARKER[f] for f in FAMILY_ORDER]),
)
"AoG scales mapping family → canonical color only (for violins, which have no marker)."
family_color_scales() = scales(
    Color = (; palette = [f => FAMILY_COLOR[f] for f in FAMILY_ORDER]),
)

const ARTICLE_METRIC_FIGURES = [
    # Primary metrics — full figure set for main article
    ("Spearman",    [:cd_benchmark, :cd_gap, :inversion, :violin_err, :violin_bench, :pvb, :pvb_per_dataset, :spread, :mahal_alg, :cd_adv, :cd_rankcor, :nemenyi_bench, :nemenyi_gap]),
    ("RMSE",        [:cd_benchmark, :cd_gap, :inversion, :violin_err, :violin_bench, :pvb, :cd_adv, :cd_rankcor]),
    ("R2",          [:cd_benchmark, :cd_gap, :inversion, :violin_err, :violin_bench, :pvb, :cd_adv, :cd_rankcor]),
    # Secondary metrics — CD diagrams only for main article
    ("MAE",         [:cd_benchmark, :cd_gap]),
    ("MedAE",       [:cd_benchmark, :cd_gap]),
    ("Pearson",     [:cd_benchmark, :cd_gap]),
    ("Kendall",     [:cd_benchmark, :cd_gap]),
    # Virtual-screening metrics — for supplementary
    ("Enrichment5",  [:cd_benchmark, :cd_gap, :violin_bench]),
    ("Enrichment10", [:cd_benchmark, :cd_gap, :violin_bench]),
    ("Enrichment20", [:cd_benchmark, :cd_gap, :violin_bench]),
    ("BEDROC",       [:cd_benchmark, :cd_gap, :violin_bench]),
]

const ARTICLE_STATIC_FIGURES = [
    ("all_metrics_heatmap",    :fig_hm),
    ("family_rank_bar",        :fig_family_bar),
    ("ad_coverage",            :fig_cov),
    ("nn_dist",                :fig_nn_dist),
    ("champ_freq",             :fig_champ_freq),
    ("intext_mahal",           :fig_intext),
    ("threshold_combined",     :fig_threshold_combined),
    ("inout_domain_spearman",  :fig_inout_domain),
]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
@info "Loading layer1 summaries..."
_raw_main  = load_layer1_summary(EXPERIMENT_ROOT; datasets=DATASETS)
_raw_clust = load_layer1_summary(EXPERIMENT_ROOT; datasets=DATASETS,
    layer1_subdir=joinpath("layer1_summary", "new_splitters"))
all_long_base = layer1_to_long(vcat(_raw_main, _raw_clust; cols=:union))

@info "Loading split metrics..."
split_metrics_avg = let _raw = load_split_metrics(EXPERIMENT_ROOT; datasets=DATASETS)
    combine(
        groupby(_raw, [:dataset, :splitter]),
        :external_mahalanobis_split_distance => mean => :mahal_dist,
        :internal_mahalanobis_split_distance => mean => :mahal_dist_internal,
        :external_Sparsity_Gap               => mean => :sparsity_gap,
    )
end  # _raw freed immediately

@info "Loading AD summary (loading $(length(DATASETS)) prediction files — may take a few minutes)..."
ad_summary = load_ad_summary(EXPERIMENT_ROOT; datasets=DATASETS)

@info "Loading champion summary..."
champ_summary = SKIP_CHAMPION ? nothing : load_champion_summary(EXPERIMENT_ROOT; datasets=DATASETS)

# ------------------------------------------------------------------
# Build analysis dataframes
# ------------------------------------------------------------------
@info "Building all_long_full and all_results..."
all_results = let _join_cols = [:dataset, :algorithm, :rep, :fold, :set],
                  _joined    = leftjoin(all_long_base, ad_summary.fold_pearson_spearman; on=_join_cols)
    merge_repeated_run_results(_joined; metrics=METRICS)
end  # _raw_main, _raw_clust, all_long_base, joined all freed
_raw_main = nothing; _raw_clust = nothing; all_long_base = nothing
GC.gc()

# Per-(algorithm, dataset, metric) averages across all metrics
all_metrics_avgs = begin
    _stacked = stack(all_results, METRICS; variable_name=:metric, value_name=:score)
    _per_set = combine(
        groupby(_stacked, [:algorithm, :dataset, :metric, :set]),
        :score => mean => :score,
    )
    combine(groupby(_per_set, [:algorithm, :dataset, :metric])) do s
        lbl  = s.set
        ips  = let v = skipmissing(s.score[lbl .== "ips"])
            isempty(v) ? missing : mean(v)
        end
        test = let v = skipmissing(s.score[lbl .== "test"])
            isempty(v) ? missing : mean(v)
        end
        m    = Symbol(s.metric[1])
        diff = (ismissing(ips) || ismissing(test)) ? missing :
               benchmark_gap(test, ips, m; higher_better=HIGHER_BETTER)
        (; ips, test, diff)
    end
end

# ------------------------------------------------------------------
# Static figure: AD coverage
# ------------------------------------------------------------------
_sp_idx = Dict(s => i for (i, s) in enumerate(SPLITTERS))

ad_coverage = ad_summary.coverage
cov_by_ds   = combine(groupby(ad_coverage, [:dataset, :splitter]),
    :coverage => mean => :mean_coverage)
_cov_plot   = copy(ad_coverage)
_cov_plot[!, :splitter_label] = [get(SPLITTER_HUMAN_MAP, s, s) for s in _cov_plot.splitter]
_cov_plot[!, :family]         = splitter_family.(_cov_plot.splitter)
sort!(_cov_plot, :splitter, by=s -> get(_sp_idx, s, 99))

fig_cov = draw(
    data(_cov_plot) *
    mapping(:splitter_label => "Splitter", :coverage => "AD Coverage";
        color=:family => "Family") *
    visual(Violin; show_median=true),
    family_color_scales();
    figure=(size=(1400, 520), fontsize=14),
    axis=(xticklabelrotation=π / 4,
        ylabel="Coverage (fraction of benchmark compounds in AD)"),
)
_save_fig("ad_coverage", fig_cov); fig_cov = nothing

# Static figure: NN distance distributions
_nn_test = copy(ad_summary.nn_test_distances)
_nn_test[!, :splitter_label] = [get(SPLITTER_HUMAN_MAP, s, s) for s in _nn_test.splitter]
_nn_test[!, :family]         = splitter_family.(_nn_test.splitter)
fig_nn_dist = draw(
    data(_nn_test) *
    mapping(:splitter_label => "Splitter",
        :nn_dist_to_train => "NN Distance to Training Set";
        color=:family => "Family") *
    visual(Violin; show_median=true,
        datalimits=(x -> (quantile(x, 0.01), quantile(x, 0.99)))),
    family_color_scales();
    figure=(size=(1400, 520), fontsize=14),
    axis=(xticklabelrotation=π / 4,
        title="NN distance from external-benchmark compounds to the training set, per splitter (1st–99th percentile)"),
)
_save_fig("nn_dist", fig_nn_dist); fig_nn_dist = nothing

# Static figure: champion model frequency (skipped when champion data not loaded)
if !SKIP_CHAMPION
    _champ = copy(champ_summary.champion_meta)
    _champ[!, :splitter_label] = [get(SPLITTER_HUMAN_MAP, s, s) for s in _champ.splitter]
    _champ_freq  = combine(groupby(_champ, [:splitter, :splitter_label, :model]), nrow => :count)
    _champ_total = combine(groupby(_champ, [:splitter]), nrow => :total)
    leftjoin!(_champ_freq, _champ_total; on=:splitter)
    _champ_freq[!, :fraction] = _champ_freq.count ./ _champ_freq.total
    sort!(_champ_freq, :splitter, by=s -> get(_sp_idx, s, 99))
    fig_champ_freq = draw(
        data(_champ_freq) *
        mapping(:splitter_label => "Splitter", :fraction => "Fraction of folds as champion";
            color=:model => "Champion model", stack=:model) *
        visual(BarPlot);
        figure=(size=(1000, 500), fontsize=12),
        axis=(xticklabelrotation=π / 4, title="Champion model frequency per splitter"),
    )
    _save_fig("champ_freq", fig_champ_freq)
    fig_champ_freq = nothing
    # champion_meta no longer needed; keep only selection_by_dataset and hitrate_by_dataset
    champ_summary = (; champ_summary.selection_by_dataset, champ_summary.hitrate_by_dataset)
    GC.gc()
end

# Static figure: internal vs external Mahalanobis
gap_vs_split = begin
    av = stack(all_results, METRICS; variable_name=:metric, value_name=:score)
    av = combine(groupby(filter(:metric => ==("Spearman"), av), [:algorithm, :dataset, :set]),
        :score => mean => :score)
    av = combine(groupby(av, [:algorithm, :dataset])) do s
        lbl  = s.set
        ips  = let v = skipmissing(s.score[lbl .== "ips"]); isempty(v) ? missing : mean(v); end
        test = let v = skipmissing(s.score[lbl .== "test"]); isempty(v) ? missing : mean(v); end
        diff = (ismissing(ips) || ismissing(test)) ? missing :
               benchmark_gap(test, ips, :Spearman; higher_better=HIGHER_BETTER)
        (; ips, test, diff)
    end
    rename(select(av, :dataset, :algorithm, :diff), :algorithm => :splitter)
end
gap_vs_split = innerjoin(gap_vs_split, split_metrics_avg; on=[:dataset, :splitter])
_mah_comp = select(dropmissing(gap_vs_split, [:mahal_dist, :mahal_dist_internal]),
    :dataset, :splitter, :mahal_dist, :mahal_dist_internal)
_mah_comp[!, :family] = splitter_family.(_mah_comp.splitter)
ρ_int_ext  = corspearman(_mah_comp.mahal_dist_internal, _mah_comp.mahal_dist)
_xlim_ie   = extrema(vcat(_mah_comp.mahal_dist_internal, _mah_comp.mahal_dist))
_diag_ie   = collect(_xlim_ie)
_fams_ie   = [f for f in FAMILY_ORDER if f in _mah_comp.family]
fig_intext = Figure(size=(850, 700), fontsize=14)
ax_ie = Axis(fig_intext[1, 1];
    title="Internal test vs external benchmark — train-set distance",
    subtitle="Diagonal: internal test mimics external difficulty  |  Spearman ρ = $(round(ρ_int_ext; digits=3))",
    xlabel="Internal test–train Mahalanobis distance",
    ylabel="External benchmark–train Mahalanobis distance",
)
for f in _fams_ie
    sub = filter(:family => ==(f), _mah_comp)
    scatter!(ax_ie, sub.mahal_dist_internal, sub.mahal_dist;
        color=(FAMILY_COLOR[f], 0.7), marker=FAMILY_MARKER[f],
        markersize=10, strokewidth=0.8, strokecolor=:black, label=f)
end
lines!(ax_ie, _diag_ie, _diag_ie; linestyle=:dash, color=:black, linewidth=1.5)
Legend(fig_intext[1, 2], ax_ie; framevisible=false)
_save_fig("intext_mahal", fig_intext); fig_intext = nothing

# Static figure: all-metrics Nemenyi heatmap
_bench_df_all = DataFrame(algorithm=String[], metric=String[], rank=Float64[])
_gap_df_all   = DataFrame(algorithm=String[], metric=String[], rank=Float64[])
for m in METRICS
    m_str = String(m)
    hb    = is_higher_better(m; higher_better=HIGHER_BETTER)
    sub   = filter(:metric => ==(m_str), all_metrics_avgs)
    isempty(sub) && continue
    nm_b, alg_ord = nemenyi_from_df(sub, :dataset, :algorithm, :ips; higher_better=hb)
    nm_g, _       = nemenyi_from_df(sub, :dataset, :algorithm, :diff; absolute=true)
    n = length(alg_ord)
    append!(_bench_df_all, DataFrame(algorithm=alg_ord, metric=fill(m_str, n), rank=nm_b.avg_ranks))
    append!(_gap_df_all,   DataFrame(algorithm=alg_ord, metric=fill(m_str, n), rank=nm_g.avg_ranks))
end
_alg_order_hm  = sort(combine(groupby(_bench_df_all, :algorithm), :rank => mean => :m), :m).algorithm
_alg_labels_hm = [get(SPLITTER_HUMAN_MAP, a, a) for a in _alg_order_hm]
_mstrs_hm      = METRICS_STRS
n_ra, n_rm     = length(_alg_order_hm), length(_mstrs_hm)
_ai_hm = Dict(a => i for (i, a) in enumerate(_alg_order_hm))
_mi_hm = Dict(m => j for (j, m) in enumerate(_mstrs_hm))
mat_b_hm = fill(NaN32, n_rm, n_ra)
mat_g_hm = fill(NaN32, n_rm, n_ra)
for r in eachrow(_bench_df_all)
    mat_b_hm[_mi_hm[r.metric], _ai_hm[r.algorithm]] = r.rank
end
for r in eachrow(_gap_df_all)
    mat_g_hm[_mi_hm[r.metric], _ai_hm[r.algorithm]] = r.rank
end
cmap_hm  = Reverse(:RdYlGn)
clims_hm = (1.0, Float64(n_ra))
fig_hm   = Figure(size=(1500, 800), fontsize=14)
for (col, mat, ttl) in ((1, mat_b_hm, "Benchmark performance rank"),
                         (2, mat_g_hm, "Gap |Δ| rank (lower = smaller estimation error)"))
    ax = Axis(fig_hm[1, col];
        title=ttl,
        xticks=(1:n_ra, _alg_labels_hm),
        yticks=(1:n_rm, [get(METRIC_DISPLAY, m, m) for m in _mstrs_hm]),
        xticklabelrotation=π / 4,
        yticklabelsvisible=(col == 1),
    )
    heatmap!(ax, 1:n_ra, 1:n_rm, mat'; colormap=cmap_hm, colorrange=clims_hm)
    for i in 1:n_ra, j in 1:n_rm
        isnan(mat[j, i]) && continue
        v = mat[j, i]
        text!(ax, i, j; text=string(round(Int, v)),
            align=(:center, :center), fontsize=10,
            color=v < n_ra / 2 ? :black : :white)
    end
end
Colorbar(fig_hm[1, 3]; colormap=cmap_hm, colorrange=clims_hm,
    label="Average rank", vertical=true, height=Relative(0.8))
_save_fig("all_metrics_heatmap", fig_hm); fig_hm = nothing

# Static figure: family-level rank bar
_fam_b = copy(_bench_df_all)
_fam_g = copy(_gap_df_all)
_fam_b[!, :family] = splitter_family.(_fam_b.algorithm)
_fam_g[!, :family] = splitter_family.(_fam_g.algorithm)
_fam_b[!, :target] .= "Benchmark"
_fam_g[!, :target] .= "Gap |Δ|"
_fam_all = combine(
    groupby(vcat(_fam_b, _fam_g), [:family, :metric, :target]),
    :rank => mean => :mean_rank,
)
fig_family_bar = draw(
    data(_fam_all) *
    mapping(:metric => "Metric", :mean_rank => "Mean average rank";
        color=:family => "Family", dodge=:family, layout=:target) *
    visual(BarPlot),
    family_color_scales();
    figure=(size=(1200, 500), fontsize=14),
    axis=(xticklabelrotation=π / 4,),
)
Label(fig_family_bar.figure[0, :],
    "Family-level mean rank across all metrics (lower = better)";
    fontsize=14, font=:bold)
_save_fig("family_rank_bar", fig_family_bar); fig_family_bar = nothing

# Static figure: threshold combined (bar chart + sensitivity sweep in one figure)
_spearman_avgs = filter(:metric => ==("Spearman"), all_metrics_avgs)
_thr_df = threshold_counts_df(_spearman_avgs, :Spearman;
    thr=0.10, higher_better=HIGHER_BETTER)
_thr_df[!, :label]  = [get(SPLITTER_HUMAN_MAP, a, a) for a in _thr_df.algorithm]
_thr_df[!, :family] = splitter_family.(_thr_df.algorithm)

fig_threshold_combined = let
    _thrs    = 0.0:0.02:0.30
    _alg_ord = _thr_df.algorithm
    _n_ds    = length(DATASETS)
    _sweep_rows = NamedTuple[]
    for alg in _alg_ord
        sub = filter(:algorithm => ==(alg), _spearman_avgs)
        for thr in _thrs
            n_exceed = count(r -> !ismissing(r.diff) && abs(r.diff) > thr, eachrow(sub))
            push!(_sweep_rows, (; algorithm=alg,
                label=get(SPLITTER_HUMAN_MAP, alg, alg),
                family=splitter_family(alg),
                threshold=Float64(thr),
                frac=n_exceed / _n_ds))
        end
    end
    _sweep_df = DataFrame(_sweep_rows)

    n   = nrow(_thr_df)
    fig = Figure(size=(2600, 700), fontsize=16)

    # Left panel: bar chart
    ax_bar = Axis(fig[1, 1];
        xticks=(1:n, _thr_df.label),
        xticklabelrotation=π / 4,
        xticklabelsize=14,
        ylabel="Number of datasets (out of 15)",
        title="Datasets with |ΔSpearman| > 0.10",
    )
    for (i, row) in enumerate(eachrow(_thr_df))
        barplot!(ax_bar, [i], [row.abs_ct]; color=FAMILY_COLOR[row.family], width=0.7)
    end
    hlines!(ax_bar, [7.5, 15.0]; linestyle=:dash, color=:gray, linewidth=1)
    xlims!(ax_bar, 0.5, n + 0.5)
    ylims!(ax_bar, 0, 15.5)

    # Right panel: sweep
    ax_sw = Axis(fig[1, 3];
        xlabel="Threshold |ΔSpearman|",
        ylabel="Fraction of datasets exceeding threshold",
        title="Threshold sensitivity sweep",
        yticks=0.0:0.2:1.0,
    )
    hlines!(ax_sw, [0.5, 1.0]; linestyle=:dot, color=(:gray, 0.5), linewidth=1)
    for alg in _alg_ord
        sub = filter(:algorithm => ==(alg), _sweep_df)
        lines!(ax_sw, sub.threshold, sub.frac;
            color=(FAMILY_COLOR[sub.family[1]], 0.8), linewidth=1.5)
    end

    # Shared legend between the two panels
    _fam_elems = [PolyElement(color=FAMILY_COLOR[f]) for f in FAMILY_ORDER if f in _thr_df.family]
    _fam_labs  = [f for f in FAMILY_ORDER if f in _thr_df.family]
    Legend(fig[1, 2], _fam_elems, _fam_labs, "Family"; framevisible=false)
    fig
end
_save_fig("threshold_combined", fig_threshold_combined); fig_threshold_combined = nothing

# Static figure: benchmark performance by applicability-domain membership
# Shows benchmark (IPS) Spearman split by whether each benchmark compound falls
# within the AD fitted on the internal training set.  Dashed line = internal
# test estimate (the potentially inflated score).  For KS-type methods the
# dashed line sits far above even the in-domain benchmark dots, quantifying
# true overoptimism; for cluster-shuffle the dashed line nearly overlaps
# with the overall benchmark level.
_bydom = copy(ad_summary.fold_metrics_by_domain)
if !isempty(_bydom) && :spearman in propertynames(_bydom)
    _bydom[!, :family] = splitter_family.(_bydom.splitter)
    _bydom[!, :domain_label] = ifelse.(_bydom.in_domain, "In-domain", "Out-of-domain")

    _dom_med = combine(
        groupby(dropmissing(_bydom, :spearman), [:splitter, :domain_label]),
        :spearman => median => :med_spearman,
        :spearman => length  => :n,
    )
    filter!(:n => >=(5), _dom_med)

    # Internal test estimate per splitter (set=="test" in all_results = internal test)
    _bench_ref = combine(
        groupby(filter(r -> r.set == "test", all_results), :algorithm),
        :Spearman => median => :bench_med,
    )
    rename!(_bench_ref, :algorithm => :splitter)

    # Order splitters by mean benchmark Spearman (IPS = benchmark in set column)
    _order_dom = sort(
        combine(groupby(_dom_med, :splitter), :med_spearman => mean => :mean_s),
        :mean_s,
    ).splitter
    _labels_dom = [get(SPLITTER_HUMAN_MAP, s, s) for s in _order_dom]
    _idx_dom    = Dict(s => i for (i, s) in enumerate(_order_dom))
    _dom_med[!, :x] = [_idx_dom[s] for s in _dom_med.splitter]

    global fig_inout_domain = Figure(size=(1400, 600), fontsize=14)
    ax_dom = Axis(fig_inout_domain[1, 1];
        xticks=(1:length(_order_dom), _labels_dom),
        xticklabelrotation=π / 4,
        ylabel="Median Spearman ρ",
    )
    # Draw connecting lines first
    for sp in _order_dom
        sub = filter(:splitter => ==(sp), _dom_med)
        nrow(sub) == 2 || continue
        xi = _idx_dom[sp]
        y_vals = sub.med_spearman
        lines!(ax_dom, [xi, xi], y_vals; color=(:gray, 0.5), linewidth=1.5)
    end
    # Draw dots
    for row in eachrow(_dom_med)
        fc = FAMILY_COLOR[splitter_family(row.splitter)]
        mk = row.domain_label == "In-domain" ? :circle : :utriangle
        scatter!(ax_dom, [row.x], [row.med_spearman];
            color=fc, marker=mk, markersize=12,
            strokewidth=0.8, strokecolor=:black)
    end
    # Internal test estimate dashed lines (set=="test")
    for row in eachrow(_bench_ref)
        haskey(_idx_dom, row.splitter) || continue
        xi = _idx_dom[row.splitter]
        linesegments!(ax_dom, [xi - 0.3, xi + 0.3], [row.bench_med, row.bench_med];
            linestyle=:dash, color=(:black, 0.5), linewidth=1.2)
    end
    # Legend
    _dom_elems = [
        MarkerElement(color=:gray,  marker=:circle,    markersize=12),
        MarkerElement(color=:gray,  marker=:utriangle, markersize=12),
        LineElement(color=(:black, 0.5), linestyle=:dash, linewidth=1.5),
    ]
    Legend(fig_inout_domain[1, 2], _dom_elems,
        ["In-domain (benchmark)", "Out-of-domain (benchmark)", "Internal test estimate"];
        framevisible=false)
else
    global fig_inout_domain = nothing
end
_save_fig("inout_domain_spearman", fig_inout_domain); fig_inout_domain = nothing
ad_summary = nothing; GC.gc()

# ------------------------------------------------------------------
# make_core_figures
# ------------------------------------------------------------------
function _core_data(metric::AbstractString)
    hb    = is_higher_better(metric; higher_better=HIGHER_BETTER)
    m_sym = Symbol(metric)

    stacked = stack(all_results, METRICS; variable_name=:_m, value_name=:_s)
    per_set = combine(
        groupby(stacked, [:algorithm, :dataset, :_m, :set]),
        :_s => mean => :_s,
    )
    avgs = combine(groupby(filter(:_m => ==(metric), per_set), [:algorithm, :dataset])) do s
        lbl  = s.set
        ips  = let v = skipmissing(s._s[lbl .== "ips"]);  isempty(v) ? missing : mean(v); end
        test = let v = skipmissing(s._s[lbl .== "test"]); isempty(v) ? missing : mean(v); end
        diff = (ismissing(ips) || ismissing(test)) ? missing :
               benchmark_gap(test, ips, m_sym; higher_better=HIGHER_BETTER)
        (; ips, test, diff)
    end
    avgs[!, :algorithm_label] = [get(SPLITTER_HUMAN_MAP, a, a) for a in avgs.algorithm]

    nm_b, alg_order = nemenyi_from_df(avgs, :dataset, :algorithm, :ips; higher_better=hb)
    nm_g, _         = nemenyi_from_df(avgs, :dataset, :algorithm, :diff; absolute=true)
    labels          = [get(SPLITTER_HUMAN_MAP, a, a) for a in alg_order]

    ranks = DataFrame(algorithm=alg_order, label=labels,
                      bench=nm_b.avg_ranks, gap=nm_g.avg_ranks)
    ranks.family = splitter_family.(ranks.algorithm)
    ranks.short  = short_code.(ranks.algorithm)

    return avgs, nm_b, nm_g, labels, ranks, alg_order
end

function _pvb_per_dataset_fig(avgs::DataFrame, metric::AbstractString)
    _ds_list   = sort(unique(avgs.dataset))
    _n_ds_cols = 3
    _n_ds_rows = cld(length(_ds_list), _n_ds_cols)
    fig_pvb_ds = Figure(size=(1200, 280 * _n_ds_rows), fontsize=11)
    Label(fig_pvb_ds[0, :], "Per-dataset performance vs bias — $metric (each point = one splitter)";
        fontsize=13, font=:bold)
    for (idx, ds) in enumerate(_ds_list)
        row_ds = div(idx - 1, _n_ds_cols) + 1
        col_ds = mod(idx - 1, _n_ds_cols) + 1
        sub_ds = filter(:dataset => ==(ds), avgs)
        sub_pvb_ds = combine(groupby(sub_ds, [:algorithm, :algorithm_label]),
            :ips  => mean => :bench,
            :diff => mean => :bias)
        sub_pvb_ds.family = splitter_family.(sub_pvb_ds.algorithm)
        ax_ds = Axis(fig_pvb_ds[row_ds, col_ds];
            title=ds, titlesize=11,
            xlabel="Benchmark $metric", ylabel="Bias Δ",
            xticklabelsize=8, yticklabelsize=8)
        hlines!(ax_ds, [0.0]; linestyle=:dash, color=:gray, linewidth=1)
        for f in [f for f in FAMILY_ORDER if f in sub_pvb_ds.family]
            s = filter(:family => ==(f), sub_pvb_ds)
            scatter!(ax_ds, s.bench, s.bias;
                color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
                markersize=8, strokewidth=0.5, strokecolor=:black)
        end
    end
    _leg_elems = [MarkerElement(color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
        markersize=10, strokewidth=0.8, strokecolor=:black) for f in FAMILY_ORDER]
    Legend(fig_pvb_ds[_n_ds_rows + 1, :], _leg_elems, FAMILY_ORDER;
        orientation=:horizontal, framevisible=false, nbanks=1)
    return fig_pvb_ds
end

function make_core_figures(metric::AbstractString)
    avgs, nm_b, nm_g, labels, ranks, alg_order = _core_data(metric)

    # CD — benchmark (skip if Nemenyi returned NaN due to degenerate data)
    cd_bench = isnan(nm_b.cd) ? nothing : cd_diagram(nm_b, labels;
        title="Benchmark performance ($metric)", width=1500)

    # CD — gap
    cd_gap = isnan(nm_g.cd) ? nothing : cd_diagram(nm_g, labels;
        title="Estimation gap |Δ| ($metric)", width=1500)

    # Nemenyi pairwise p-value heatmaps (Spearman only; skip for other metrics)
    fig_nemenyi_bench = (metric == "Spearman" && !isnan(nm_b.cd)) ?
        nemenyi_heatmap(nm_b, labels; alpha=0.05) : nothing
    fig_nemenyi_gap   = (metric == "Spearman" && !isnan(nm_g.cd)) ?
        nemenyi_heatmap(nm_g, labels; alpha=0.05) : nothing

    # Inversion scatter
    ρ_r   = corspearman(ranks.bench, ranks.gap)
    plt_i = data(ranks) *
            mapping(:bench => "Benchmark rank (1=best)", :gap => "Gap rank (1=best)",
                    color=:family => "Family", marker=:family => "Family") *
            visual(Scatter; markersize=14, strokewidth=1.2, strokecolor=:black)
    inv_f = draw(plt_i, family_cm_scales();
        figure=(size=(1100, 550), fontsize=14),
        axis=(title="Performance–estimation inversion",
              subtitle="Rank correlation: ρ = $(round(ρ_r; digits=2))",
              xreversed=true, yreversed=true, xlabelsize=15, ylabelsize=15, titlesize=18))
    lo_r, hi_r = minimum(vcat(ranks.bench, ranks.gap)), maximum(vcat(ranks.bench, ranks.gap))
    lines!(content(inv_f.figure[1,1]), [lo_r,hi_r], [lo_r,hi_r]; linestyle=:dash, linewidth=2, color=:gray)
    annotation!(content(inv_f.figure[1,1]), Point2f.(ranks.bench, ranks.gap);
        text=String.(ranks.short), fontsize=10, textcolor=:black, labelspace=:relative_pixel)

    # Violin — estimation bias
    avgs_e = filter(r -> !ismissing(r.diff) && isfinite(r.diff), copy(avgs))
    _oe = sortperm(combine(groupby(avgs_e, :algorithm), :diff => median => :m).m)
    _le = combine(groupby(avgs_e, :algorithm), :diff => median => :m).algorithm[_oe]
    avgs_e.algorithm_label = categorical(
        avgs_e.algorithm_label; ordered=true,
        levels=[get(SPLITTER_HUMAN_MAP, a, a) for a in _le])
    avgs_e[!, :family] = splitter_family.(avgs_e.algorithm)
    viol_err = draw(
        data(avgs_e) * mapping(:algorithm_label, :diff; color=:family => "Family") *
        visual(Violin; show_median=true),
        family_color_scales();
        figure=(size=(PLOT_WIDTH, 750), fontsize=14),
        axis=(xticklabelrotation=π / 3, title="Estimation bias (Δ)",
              ylabel="Signed bias Δ = internal test − external benchmark",
              xlabel="Splitting method"))
    # Overlay per-dataset median dots so readers see between-dataset spread
    let _lvl_e = levels(avgs_e.algorithm_label),
        _pos_e = Dict(string(l) => i for (i, l) in enumerate(_lvl_e)),
        _dsm_e = combine(groupby(avgs_e, [:algorithm_label, :dataset]), :diff => median => :m)
        _xe = [get(_pos_e, string(lbl), 0) for lbl in _dsm_e.algorithm_label]
        scatter!(content(viol_err.figure[1, 1]), Float64.(_xe), _dsm_e.m;
            color=(:black, 0.35), markersize=5, marker=:circle)
    end

    # Violin — benchmark performance
    avgs_b = filter(r -> !ismissing(r.ips) && isfinite(r.ips), copy(avgs))
    _ob = sortperm(combine(groupby(avgs_b, :algorithm), :ips => median => :m).m)
    _lb = combine(groupby(avgs_b, :algorithm), :ips => median => :m).algorithm[_ob]
    avgs_b.algorithm_label = categorical(
        avgs_b.algorithm_label; ordered=true,
        levels=[get(SPLITTER_HUMAN_MAP, a, a) for a in _lb])
    avgs_b[!, :family] = splitter_family.(avgs_b.algorithm)
    viol_bench = draw(
        data(avgs_b) * mapping(:algorithm_label, :ips; color=:family => "Family") *
        visual(Violin; show_median=true),
        family_color_scales();
        figure=(size=(PLOT_WIDTH, 750), fontsize=14),
        axis=(xticklabelrotation=π / 3, title="External benchmark performance",
              ylabel="External benchmark $metric", xlabel="Splitting method"))
    let _lvl_b = levels(avgs_b.algorithm_label),
        _pos_b = Dict(string(l) => i for (i, l) in enumerate(_lvl_b)),
        _dsm_b = combine(groupby(avgs_b, [:algorithm_label, :dataset]), :ips => median => :m)
        _xb = [get(_pos_b, string(lbl), 0) for lbl in _dsm_b.algorithm_label]
        scatter!(content(viol_bench.figure[1, 1]), Float64.(_xb), _dsm_b.m;
            color=(:black, 0.35), markersize=5, marker=:circle)
    end

    # Performance vs bias scatter
    pvb = combine(groupby(avgs, [:algorithm, :algorithm_label]),
        :ips => median => :bench_med, :diff => median => :bias_med)
    pvb[!, :family] = splitter_family.(pvb.algorithm)
    pvb[!, :short]  = short_code.(pvb.algorithm)
    fams_p  = [f for f in FAMILY_ORDER if f in pvb.family]
    fig_pvb = Figure(size=(900, 600), fontsize=14)
    ax_pvb  = Axis(fig_pvb[1, 1];
        xlabel="Median external benchmark $metric",
        ylabel="Median signed bias Δ = internal test − benchmark",
        title="Performance vs estimation bias — $metric",
        subtitle="Each point = one splitter; dashed line = zero bias (positive = optimistic)")
    hlines!(ax_pvb, [0.0]; linestyle=:dash, color=:gray, linewidth=1.5)
    for f in fams_p
        sub = filter(:family => ==(f), pvb)
        scatter!(ax_pvb, sub.bench_med, sub.bias_med;
            color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f],
            markersize=15, strokewidth=1.2, strokecolor=:black, label=f)
    end
    annotation!(ax_pvb, Point2f.(pvb.bench_med, pvb.bias_med);
        text=String.(pvb.short), fontsize=10, textcolor=:black, labelspace=:relative_pixel)
    Legend(fig_pvb[1, 2], ax_pvb; framevisible=false)

    # Per-dataset performance vs bias grid (5×3 small multiples, one panel per dataset)
    fig_pvb_ds = _pvb_per_dataset_fig(avgs, metric)

    # Per-dataset spread
    spread = plot_per_dataset_spread(avgs; metric_label=metric,
        family_colors=FAMILY_COLOR, family_markers=FAMILY_MARKER)

    # Mahalanobis: per-algorithm scatter
    gap_mah = let av = rename(select(avgs, :dataset, :algorithm, :diff), :algorithm => :splitter)
        innerjoin(av, split_metrics_avg; on=[:dataset, :splitter])
    end
    gap_mah.family = splitter_family.(gap_mah.splitter)
    mah_sum = combine(groupby(gap_mah, :splitter),
        :mahal_dist => median => :mahal_med,
        :diff => (x -> median(abs.(x))) => :gap_med)
    mah_sum.family = splitter_family.(mah_sum.splitter)
    mah_sum.short  = short_code.(mah_sum.splitter)
    ρ_alg   = corspearman(mah_sum.mahal_med, mah_sum.gap_med)
    fig_mah = draw(
        data(mah_sum) *
        mapping(:mahal_med => "Median Mahalanobis distance",
                :gap_med   => "Median |Δ| bias ($metric)",
                color=:family => "Family", marker=:family => "Family") *
        visual(Scatter; markersize=15, strokewidth=1.2, strokecolor=:black),
        family_cm_scales();
        figure=(size=(900, 550), fontsize=14),
        axis=(title="Per-algorithm: split quality vs estimation bias",
              subtitle="Spearman ρ = $(round(ρ_alg; digits=3))", titlesize=16))
    annotation!(content(fig_mah.figure[1,1]), Point2f.(mah_sum.mahal_med, mah_sum.gap_med);
        text=String.(mah_sum.short), fontsize=11, textcolor=:black, labelspace=:relative_pixel)

    # Champion CD diagrams (skipped when champion data not loaded)
    _met_lc  = lowercase(metric)
    _adv_col = Symbol("best_adv_$_met_lc")
    _rk_col  = Symbol("hitrate_$_met_lc")
    _sel_ds  = SKIP_CHAMPION ? nothing : champ_summary.selection_by_dataset
    _rk_ds   = SKIP_CHAMPION ? nothing : champ_summary.hitrate_by_dataset

    cd_adv = if !SKIP_CHAMPION && _adv_col in propertynames(_sel_ds)
        _nm_a, _ord_a = nemenyi_from_df(_sel_ds, :dataset, :splitter, _adv_col; higher_better=true)
        _lbl_a = [get(SPLITTER_HUMAN_MAP, s, s) for s in _ord_a]
        isnan(_nm_a.cd) ? nothing : cd_diagram(_nm_a, _lbl_a; title="CD — Selection advantage ($metric)", width=2000, height=650, legend_labelsize=26)
    else
        nothing
    end

    cd_rankcor = if !SKIP_CHAMPION && _rk_col in propertynames(_rk_ds)
        _nm_r, _ord_r = nemenyi_from_df(_rk_ds, :dataset, :splitter, _rk_col; higher_better=true)
        _lbl_r = [get(SPLITTER_HUMAN_MAP, s, s) for s in _ord_r]
        isnan(_nm_r.cd) ? nothing : cd_diagram(_nm_r, _lbl_r; title="CD — Selection hit rate ($metric)", width=2000, height=650, legend_labelsize=26)
    else
        nothing
    end

    return (;
        cd_benchmark   = cd_bench,
        cd_gap         = cd_gap,
        inversion      = inv_f,
        violin_err     = viol_err,
        violin_bench   = viol_bench,
        pvb            = fig_pvb,
        pvb_per_dataset = fig_pvb_ds,
        spread         = spread,
        mahal_alg      = fig_mah,
        cd_adv         = cd_adv,
        cd_rankcor     = cd_rankcor,
        nemenyi_bench  = fig_nemenyi_bench,
        nemenyi_gap    = fig_nemenyi_gap,
    )
end

# ------------------------------------------------------------------
# Save metric-dependent figures
# ------------------------------------------------------------------
GC.gc()
@info "Generating and saving metric-dependent figures..."
for (metric, keys) in ARTICLE_METRIC_FIGURES
    @info "  metric=$metric"
    figs = make_core_figures(metric)
    for k in keys
        fig = getfield(figs, k)
        isnothing(fig) && continue
        fname = joinpath(OUTDIR, "$(metric)_$(k).pdf")
        save(fname, fig)
    end
    GC.gc()
end

# Static figures were already saved immediately after creation above.

# Combined figures assembled with ImageMagick (+append = horizontal stitch)
@info "Creating combined figures..."
function _combine_h(dest, srcs...)
    missing_srcs = filter(!isfile, srcs)
    if !isempty(missing_srcs)
        @warn "Skipping $dest: missing $(join(missing_srcs, ", "))"
        return
    end
    run(`magick -density 300 $srcs +append -compress lzw $dest`)
    @info "  saved $dest"
end

_combine_h(
    joinpath(OUTDIR, "champ_combined.tiff"),
    joinpath(OUTDIR, "Spearman_cd_adv.pdf"),
    joinpath(OUTDIR, "Spearman_cd_rankcor.pdf"),
)
_combine_h(
    joinpath(OUTDIR, "cd_bench_combined.tiff"),
    joinpath(OUTDIR, "RMSE_cd_benchmark.pdf"),
    joinpath(OUTDIR, "R2_cd_benchmark.pdf"),
)
_combine_h(
    joinpath(OUTDIR, "cd_gap_combined.tiff"),
    joinpath(OUTDIR, "RMSE_cd_gap.pdf"),
    joinpath(OUTDIR, "R2_cd_gap.pdf"),
)
_combine_h(
    joinpath(OUTDIR, "inversion_combined.tiff"),
    joinpath(OUTDIR, "Spearman_inversion.pdf"),
    joinpath(OUTDIR, "R2_inversion.pdf"),
    joinpath(OUTDIR, "RMSE_inversion.pdf"),
)

@info "Done. Figures saved to $OUTDIR"
