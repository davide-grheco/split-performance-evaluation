using Printf
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
using DataSplitBench, DataFrames, Statistics

EXPERIMENT_ROOT = get(ENV, "EXPERIMENT_ROOT",
    joinpath(@__DIR__, "..", "experiments", "revision_v1"))

DATASETS = ["3A4", "CB1", "DPP4", "HIVINT", "HIVPROT", "LOGD", "METAB",
    "NK1", "OX1", "OX2", "PGP", "PPB", "RAT_F", "TDI", "THROMBIN"]

function cohens_d(a, b)
    a, b = collect(skipmissing(a)), collect(skipmissing(b))
    n_a, n_b = length(a), length(b)
    (n_a < 2 || n_b < 2) && return NaN
    pooled_sd = sqrt(((n_a - 1) * var(a) + (n_b - 1) * var(b)) / (n_a + n_b - 2))
    pooled_sd ≈ 0 && return NaN
    (mean(a) - mean(b)) / pooled_sd
end

@info "Loading champion summary (≈10 min)..."
cs = DataSplitBench.load_champion_summary(EXPERIMENT_ROOT; datasets=DATASETS)

# ── Hit rate ──────────────────────────────────────────────────────────────────
println("\n=== Selection hit rate (Spearman) — fraction of folds where internally selected model is also externally best ===")
hr = combine(groupby(cs.hitrate_by_dataset, :splitter),
    :hitrate_spearman => (x -> mean(skipmissing(x))) => :mean,
    :hitrate_spearman => (x -> std(skipmissing(x))) => :sd,
    :hitrate_spearman => (x -> median(skipmissing(x))) => :median)
sort!(hr, :mean; rev=true)
println("  $(rpad("Splitter", 35)) $(lpad("mean", 6))  $(lpad("sd", 6))  $(lpad("median", 7))")
for r in eachrow(hr)
    @printf("  %-35s  %.3f   %.3f   %.3f\n", r.splitter, r.mean, r.sd, r.median)
end

println("\n=== Selection advantage (Spearman) — champion external Spearman minus best alternative's ===")
sa = combine(groupby(cs.selection_by_dataset, :splitter),
    :best_adv_spearman => (x -> mean(skipmissing(x))) => :mean,
    :best_adv_spearman => (x -> std(skipmissing(x))) => :sd,
    :best_adv_spearman => (x -> median(skipmissing(x))) => :median)
sort!(sa, :mean; rev=true)
println("  $(rpad("Splitter", 35)) $(lpad("mean", 7))  $(lpad("sd", 6))  $(lpad("median", 8))")
for r in eachrow(sa)
    @printf("  %-35s  %+.4f   %.4f   %+.4f\n", r.splitter, r.mean, r.sd, r.median)
end

println("\n=== Effect sizes vs SPXY (best) and k-means-shuffle (worst) ===")
println("\n  Cohen's d for hit rate (SPXY as reference, positive = SPXY better):")
spxy_hr_vals = filter(:splitter => ==("spxy-jaccard"), cs.hitrate_by_dataset).hitrate_spearman
kmeans_hr_vals = filter(:splitter => ==("kmeans_shuffle"), cs.hitrate_by_dataset).hitrate_spearman
hr2 = sort(hr, :mean; rev=true)
for r in eachrow(hr2)
    vals = filter(:splitter => ==(r.splitter), cs.hitrate_by_dataset).hitrate_spearman
    d_vs_spxy = cohens_d(spxy_hr_vals, vals)
    d_vs_kmeans = cohens_d(vals, kmeans_hr_vals)
    @printf("  %-35s  d_vs_SPXY=%+.2f   d_vs_kmeans-shuf=%+.2f\n",
        r.splitter, d_vs_spxy, d_vs_kmeans)
end

println("\n  Cohen's d for selection advantage (SPXY as reference, positive = SPXY better):")
spxy_adv_vals = filter(:splitter => ==("spxy-jaccard"), cs.selection_by_dataset).best_adv_spearman
kmeans_adv_vals = filter(:splitter => ==("kmeans_shuffle"), cs.selection_by_dataset).best_adv_spearman
sa2 = sort(sa, :mean; rev=true)
for r in eachrow(sa2)
    vals = filter(:splitter => ==(r.splitter), cs.selection_by_dataset).best_adv_spearman
    d_vs_spxy = cohens_d(spxy_adv_vals, vals)
    d_vs_kmeans = cohens_d(vals, kmeans_adv_vals)
    @printf("  %-35s  d_vs_SPXY=%+.2f   d_vs_kmeans-shuf=%+.2f\n",
        r.splitter, d_vs_spxy, d_vs_kmeans)
end

println("\n=== SPXY vs k-means-shuffle detail ===")
for (label, sp, km, col) in [
    ("Hit rate", spxy_hr_vals, kmeans_hr_vals, ""),
    ("Selection advantage", spxy_adv_vals, kmeans_adv_vals, ""),
]
    sp_v = collect(skipmissing(sp))
    km_v = collect(skipmissing(km))
    d = cohens_d(sp, km)
    println("\n  $label:")
    @printf("    SPXY:         mean=%+.4f  sd=%.4f  median=%+.4f  min=%+.4f  max=%+.4f\n",
        mean(sp_v), std(sp_v), median(sp_v), minimum(sp_v), maximum(sp_v))
    @printf("    k-means-shuf: mean=%+.4f  sd=%.4f  median=%+.4f  min=%+.4f  max=%+.4f\n",
        mean(km_v), std(km_v), median(km_v), minimum(km_v), maximum(km_v))
    @printf("    Cohen's d (SPXY − k-means-shuf): %.2f\n", d)
end

println("\nDone.")
