# Robustness check (source of Supplementary Table S1): compare the Spearman
# benchmark performance, signed estimation bias, and performance–estimation
# inversion computed from the fixed-model splitting-assessment experiment
# (LightGBM) vs the per-split TPE-selected model of the model-selection experiment.
#
# Explicit semantics: set=="ips" is the internal test split; set=="test" is the
# external temporal benchmark. Benchmark performance = Spearman on "test".
# Estimation bias Δ = Spearman(ips) − Spearman(test)  (>0 = optimistic).
#
# Run: julia --project=. scripts/prototype_l2_inversion.jl
using Arrow, DataFrames, Statistics, StatsBase
const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "Friedman.jl"))
include(joinpath(SRC, "Nemenyi.jl"))
include(joinpath(SRC, "AnalysisUtils.jl"))
include(joinpath(SRC, "ExperimentLoader.jl"))

const ROOT = joinpath(@__DIR__, "..", "experiments", "revision_v1")
const DATASETS = ["3A4","CB1","DPP4","HIVINT","HIVPROT","LOGD","METAB","NK1",
                  "OX1","OX2","PGP","PPB","RAT_F","TDI","THROMBIN"]

"per-(dataset,splitter) benchmark + gap, per-dataset ranks, and inversion ρ.
`long` columns: dataset, splitter, rep, fold, set, spearman."
function inversion_stats(long::DataFrame)
    long = combine(groupby(long, [:dataset,:splitter,:rep,:fold,:set]),
                   :spearman => (x->mean(skipmissing(x))) => :spearman)
    w = unstack(long, [:dataset,:splitter,:rep,:fold], :set, :spearman)
    ("ips" in names(w) && "test" in names(w)) || error("missing ips/test columns: $(names(w))")
    w = dropmissing(w, ["ips","test"])
    # EMPIRICAL: ips = external benchmark, test = internal split (opposite of the
    # loader doc-comment; verified against article numbers). Optimism Δ = internal − benchmark.
    w.gap = w.test .- w.ips
    agg = combine(groupby(w, [:dataset,:splitter]),
        "ips" => mean => :bench,
        :gap   => (x->mean(abs.(x))) => :absgap,
        :gap   => mean => :signed)
    agg = transform(groupby(agg, :dataset),
        :bench  => (x->tiedrank(-x)) => :bench_rank,   # higher bench → rank 1
        :absgap => (x->tiedrank(x))  => :gap_rank)      # lower |gap| → rank 1
    mr = combine(groupby(agg, :splitter),
        :bench_rank => mean => :bench_rank,
        :gap_rank   => mean => :gap_rank,
        :bench      => mean => :bench,
        :signed     => mean => :signed)
    sort!(mr, :bench_rank)
    ρ = corspearman(mr.bench_rank, mr.gap_rank)
    return mr, ρ
end

# ---- Layer 1: fixed LightGBM -------------------------------------------------
@info "Loading fixed-model (LightGBM) predictions..."
ad = load_ad_summary(ROOT; datasets=DATASETS)
l1 = select(ad.fold_pearson_spearman, :dataset, :algorithm => :splitter, :rep, :fold, :set,
            :Spearman => :spearman)
mr1, ρ1 = inversion_stats(l1)

# ---- Layer 2: TPE-selected champion -----------------------------------------
@info "Loading selected-model (champion) predictions per dataset..."
acc = DataFrame[]
for ds in DATASETS
    raw = load_ad_champion(ROOT; datasets=[ds],
        select_cols=[:dataset,:splitter,:rep,:fold,:set,:model,:is_champion,:y_true,:y_pred])
    isempty(raw) && (@warn "no champion data" ds; continue)
    champ = filter(:is_champion => x->coalesce(x,false), dropmissing(raw, [:y_true,:y_pred]))
    isempty(champ) && (@warn "no champion rows" ds; continue)
    m = compute_fold_metrics(champ; group_cols=[:dataset,:splitter,:rep,:fold,:set])
    push!(acc, select(m, :dataset,:splitter,:rep,:fold,:set, :spearman))
    GC.gc()
end
l2 = vcat(acc...; cols=:union)
mr2, ρ2 = inversion_stats(l2)

# ---- Report ------------------------------------------------------------------
println("\n================ SPLITTER COMPARISON (mean over 15 datasets) ================")
cmp = outerjoin(
    select(mr1, :splitter, :bench=>:L1_bench, :signed=>:L1_signedΔ, :bench_rank=>:L1_brank, :gap_rank=>:L1_grank),
    select(mr2, :splitter, :bench=>:L2_bench, :signed=>:L2_signedΔ, :bench_rank=>:L2_brank, :gap_rank=>:L2_grank);
    on=:splitter)
sort!(cmp, :L1_brank)
for r in eachrow(cmp)
    println(rpad(r.splitter, 24),
        "  L1 bench=", rpad(round(r.L1_bench;digits=3),6), " Δ=", rpad(round(r.L1_signedΔ;digits=3),7),
        " | L2 bench=", rpad(round(coalesce(r.L2_bench,NaN);digits=3),6), " Δ=", rpad(round(coalesce(r.L2_signedΔ,NaN);digits=3),7))
end
println("\nInversion ρ (benchmark-rank vs gap-rank):")
println("  Fixed model  (Layer 1): ρ = ", round(ρ1; digits=3), "   [article reports −0.73]")
println("  Selected model (Layer 2): ρ = ", round(ρ2; digits=3))
println("\nUnique splitters — L1: ", sort(unique(mr1.splitter)))
println("Unique splitters — L2: ", sort(unique(mr2.splitter)))
