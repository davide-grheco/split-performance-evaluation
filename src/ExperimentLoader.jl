using Arrow
using DataFrames
using PooledArrays
using Statistics

const _AD_CAT_COLS   = (:dataset, :splitter, :set, :model)
const _AD_FLOAT_COLS = (:y_true, :y_pred, :ad_score, :nn_dist_to_train)

function _compact_ad_df!(df::DataFrame)
    for col in _AD_CAT_COLS
        col in propertynames(df) && (df[!, col] = PooledArray(df[!, col]))
    end
    for col in _AD_FLOAT_COLS
        col in propertynames(df) || continue
        v = df[!, col]
        df[!, col] = eltype(v) >: Missing ?
            passmissing(Float32).(v) : Float32.(v)
    end
    df
end

export load_layer1_summary,
    load_split_metrics,
    layer1_to_long,
    load_ad_predictions,
    load_ad_champion,
    load_champion_meta,
    load_ad_summary,
    load_champion_summary,
    LAYER1_METRICS,
    LAYER1_HIGHER_BETTER

const LAYER1_METRICS = [:R2, :RMSE, :MAE, :Bias]

const LAYER1_HIGHER_BETTER = Dict(
    :R2   => true,
    :RMSE => false,
    :MAE  => false,
    :Bias => false,
)

"""
    load_layer1_summary(experiment_root; datasets, models, layer1_subdir) -> DataFrame

Load and concatenate layer1 summary Arrow files from
`experiment_root/<layer1_subdir>/<dataset>/<model>.arrow`.

`layer1_subdir` defaults to `"layer1_summary"`.  Pass e.g.
`joinpath("layer1_summary", "new_splitters")` to load from a subdirectory
that holds additional splitter results stored separately.
"""
function load_layer1_summary(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
    models::AbstractVector{<:AbstractString}=["lightgbm"],
    layer1_subdir::AbstractString="layer1_summary",
)
    dfs = DataFrame[]
    for ds in datasets, model in models
        path = joinpath(experiment_root, layer1_subdir, ds, "$model.arrow")
        if !isfile(path)
            @warn "layer1_summary not found" dataset=ds model=model path=path
            continue
        end
        push!(dfs, DataFrame(Arrow.Table(path)))
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

"""
    load_split_metrics(experiment_root; datasets) -> DataFrame

Load and concatenate split metrics Arrow files from `experiment_root/split_metrics/`.
"""
function load_split_metrics(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
)
    dfs = DataFrame[]
    for ds in datasets
        path = joinpath(experiment_root, "split_metrics", "$ds.arrow")
        if !isfile(path)
            @warn "split_metrics not found" dataset=ds path=path
            continue
        end
        push!(dfs, DataFrame(Arrow.Table(path)))
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

"""
    load_ad_predictions(experiment_root; datasets, models) -> DataFrame

Load and concatenate AD-annotated prediction Arrow files from
`experiment_root/ad_predictions/<dataset>/<model>.arrow`.

Columns: dataset, splitter, ratio, rep, fold, set, sample_id, model,
is_champion, y_true, y_pred, ad_score, in_domain, nn_dist_to_train.
"""
function load_ad_predictions(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
    models::AbstractVector{<:AbstractString}=["lightgbm"],
)
    dfs = DataFrame[]
    for ds in datasets, model in models
        path = joinpath(experiment_root, "ad_predictions", ds, "$model.arrow")
        if !isfile(path)
            @warn "ad_predictions not found" dataset=ds model=model path=path
            continue
        end
        push!(dfs, _compact_ad_df!(DataFrame(Arrow.Table(path))))
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

"""
    load_ad_champion(experiment_root; datasets, select_cols) -> DataFrame

Load AD-annotated predictions for all candidate models from
`experiment_root/ad_champion/<dataset>.arrow`.

Same schema as `load_ad_predictions` but covers all competing models
(not only the champion), enabling cross-model performance comparisons.

`select_cols` restricts which columns are materialised.  Pass `nothing`
(default) to load all columns, or a vector of Symbols to load a subset
(reduces memory for large datasets).

Columns: dataset, splitter, ratio, rep, fold, set, sample_id, model,
is_champion, y_true, y_pred, ad_score, in_domain, nn_dist_to_train.
"""
function load_ad_champion(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
    select_cols::Union{Nothing, AbstractVector{Symbol}} = nothing,
)
    dfs = DataFrame[]
    for ds in datasets
        path = joinpath(experiment_root, "ad_champion", "$ds.arrow")
        if !isfile(path)
            @warn "ad_champion not found" dataset=ds path=path
            continue
        end
        t  = Arrow.Table(path)
        df = if isnothing(select_cols)
            DataFrame(t)
        else
            available = [c for c in select_cols if c in keys(t)]
            DataFrame([c => t[c] for c in available]; copycols=true)
        end
        push!(dfs, _compact_ad_df!(df))
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

"""
    load_champion_meta(experiment_root; datasets) -> DataFrame

Load champion-model selection metadata from
`experiment_root/champion_predictions/<dataset>.arrow`.

Only meta columns are materialised (y_true/y_pred are skipped) and rows
are deduplicated to one entry per (dataset, splitter, ratio, rep, fold, model).

Columns: dataset, splitter, ratio, rep, fold, model, is_champion, cv_rms,
round_eliminated.
"""
function load_champion_meta(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
)
    meta_cols = [:dataset, :splitter, :ratio, :rep, :fold,
                 :model, :is_champion, :cv_rms, :round_eliminated]
    dfs = DataFrame[]
    for ds in datasets
        path = joinpath(experiment_root, "champion_predictions", "$ds.arrow")
        if !isfile(path)
            @warn "champion_predictions not found" dataset=ds path=path
            continue
        end
        t         = Arrow.Table(path)
        available = [c for c in meta_cols if c in keys(t)]
        df        = DataFrame([c => t[c] for c in available]; copycols=true)
        unique!(df)
        push!(dfs, df)
    end
    isempty(dfs) ? DataFrame() : vcat(dfs...; cols=:union)
end

"""
    load_ad_summary(experiment_root; datasets, models) -> NamedTuple

Load AD-annotated raw predictions one dataset at a time, compute all per-fold
summaries, and return only the small aggregated DataFrames.  Raw predictions for
each dataset are freed before loading the next, keeping peak memory proportional
to the largest single dataset rather than the full corpus.

Returned fields:
- `fold_pearson_spearman` : per-(dataset, algorithm, rep, fold, set) Pearson + Spearman
- `coverage`              : per-(dataset, splitter, rep, fold) fraction of IPS samples in-domain
- `fold_metrics_by_domain`: per-(dataset, splitter, rep, fold, in_domain) RMSE/Pearson/Spearman on IPS
- `fold_metrics_in_domain`: per-(dataset, splitter, rep, fold) metrics for in-domain test samples (gap analysis)
- `nn_test_distances`     : (dataset, splitter, nn_dist_to_train) for IPS rows only
"""
function load_ad_summary(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
    models::AbstractVector{<:AbstractString}=["lightgbm"],
)
    fold_ps_acc        = DataFrame[]
    coverage_acc       = DataFrame[]
    fold_by_domain_acc = DataFrame[]
    fold_in_domain_acc = DataFrame[]
    nn_dists_acc       = DataFrame[]

    for ds in datasets
        raw = load_ad_predictions(experiment_root; datasets=[ds], models=models)
        isempty(raw) && continue

        push!(fold_ps_acc, compute_fold_metrics(
            dropmissing(raw, [:y_true, :y_pred]);
            group_cols=[:dataset, :splitter, :rep, :fold, :set]))

        push!(coverage_acc, compute_ad_coverage(raw))

        ips_ad = filter(r -> r.set == "ips" && !ismissing(r.in_domain), raw)
        push!(fold_by_domain_acc, compute_fold_metrics(ips_ad;
            group_cols=[:dataset, :splitter, :rep, :fold, :in_domain]))

        test_in = filter(r -> r.set == "test" && coalesce(r.in_domain, false), raw)
        push!(fold_in_domain_acc, compute_fold_metrics(test_in;
            group_cols=[:dataset, :splitter, :rep, :fold]))

        push!(nn_dists_acc, select(
            filter(r -> r.set == "ips" && !ismissing(r.nn_dist_to_train), raw),
            :dataset, :splitter, :nn_dist_to_train))
        GC.gc()   # free `raw` before loading the next (potentially large) dataset
    end

    fold_ps = vcat(fold_ps_acc...; cols=:union)
    rename!(fold_ps, :splitter => :algorithm,
        :pearson       => :Pearson,
        :spearman      => :Spearman,
        :kendall       => :Kendall,
        :medae         => :MedAE,
        :enrichment_5  => :Enrichment5,
        :enrichment_10 => :Enrichment10,
        :enrichment_20 => :Enrichment20,
        :bedroc        => :BEDROC)
    select!(fold_ps, :dataset, :algorithm, :rep, :fold, :set,
        :Pearson, :Spearman, :Kendall, :MedAE, :Enrichment5, :Enrichment10, :Enrichment20, :BEDROC)

    return (;
        fold_pearson_spearman  = fold_ps,
        coverage               = vcat(coverage_acc...; cols=:union),
        fold_metrics_by_domain = vcat(fold_by_domain_acc...; cols=:union),
        fold_metrics_in_domain = vcat(fold_in_domain_acc...; cols=:union),
        nn_test_distances      = vcat(nn_dists_acc...; cols=:union),
    )
end

"""
    load_champion_summary(experiment_root; datasets) -> NamedTuple

Load champion-model AD predictions and metadata one dataset at a time, compute
all needed summaries, and return only the aggregated DataFrames.  Raw predictions
for each dataset are freed before loading the next.

Returned fields:
- `champion_meta`        : filtered champion-only rows (model frequency + CV RMSE charts)
- `ips_advantage`        : per-(dataset, splitter) mean in-domain IPS RMSE advantage
- `fold_selection`       : per-(dataset, splitter, rep, fold) champion selection quality
- `selection_by_dataset` : per-(dataset, splitter) mean advantage aggregated for Nemenyi
- `hitrate_by_dataset`   : per-(dataset, splitter) fraction of folds where champion is externally best
"""
# Minimal columns needed for champion-summary metrics (avoids loading
# nn_dist_to_train, ad_score, sample_id, ratio which are not used here).
const _CHAMP_SUMMARY_COLS = [:dataset, :splitter, :rep, :fold, :set,
                              :model, :is_champion, :y_true, :y_pred, :in_domain]

function load_champion_summary(
    experiment_root::AbstractString;
    datasets::AbstractVector{<:AbstractString},
)
    champ_only_acc  = DataFrame[]
    ips_fold_adv_acc = DataFrame[]
    fold_sel_acc    = DataFrame[]

    for ds in datasets
        ad_champ = load_ad_champion(experiment_root; datasets=[ds],
                                    select_cols=_CHAMP_SUMMARY_COLS)
        meta     = load_champion_meta(experiment_root; datasets=[ds])
        (isempty(ad_champ) || isempty(meta)) && continue

        co = filter(:is_champion => x -> coalesce(x, false), meta)
        unique!(co, [:dataset, :splitter, :ratio, :rep, :fold])
        push!(champ_only_acc, co)

        ips_id = filter(r -> r.set == "ips" && coalesce(r.in_domain, false), ad_champ)
        ips_metrics = compute_fold_metrics(ips_id;
            group_cols=[:dataset, :splitter, :rep, :fold, :model, :is_champion])
        ips_fold_adv = combine(groupby(ips_metrics, [:dataset, :splitter, :rep, :fold])) do g
            n = nrow(g)
            ci = findfirst(x -> coalesce(x, false), g.is_champion)
            (isnothing(ci) || n < 2) && return (; ips_adv_rmse=missing)
            disc = setdiff(1:n, [ci])
            (; ips_adv_rmse=minimum(g.rmse[disc]) - g.rmse[ci])
        end
        push!(ips_fold_adv_acc, ips_fold_adv)

        bench_metrics = compute_fold_metrics(
            filter(:set => ==("ips"), ad_champ);
            group_cols=[:dataset, :splitter, :rep, :fold, :model, :is_champion])
        leftjoin!(bench_metrics,
            unique(select(meta, :dataset, :splitter, :rep, :fold, :model, :cv_rms));
            on=[:dataset, :splitter, :rep, :fold, :model])
        push!(fold_sel_acc, compute_fold_selection_quality(bench_metrics))
        GC.gc()   # free ad_champ before loading the next (potentially large) dataset
    end

    champ_only   = vcat(champ_only_acc...; cols=:union)
    ips_fold_adv = vcat(ips_fold_adv_acc...; cols=:union)
    fold_sel     = vcat(fold_sel_acc...; cols=:union)

    ips_adv_ds = combine(
        groupby(dropmissing(ips_fold_adv, :ips_adv_rmse), [:dataset, :splitter]),
        :ips_adv_rmse => mean => :ips_adv_rmse,
    )
    sel_ds, rk_ds = aggregate_selection_by_dataset(fold_sel)

    return (;
        champion_meta        = champ_only,
        ips_advantage        = ips_adv_ds,
        fold_selection       = fold_sel,
        selection_by_dataset = sel_ds,
        hitrate_by_dataset   = rk_ds,
    )
end

"""
    layer1_to_long(df) -> DataFrame

Convert wide-format layer1 summary (columns `r2_ips`, `r2_test`, `rmse_ips`, …)
to the long format used by the analysis pipeline:
`[:dataset, :algorithm, :ratio, :model, :rep, :fold, :set, :R2, :RMSE, :MAE, :Bias]`.

The `set` column takes values `"ips"` or `"test"`. NOTE the counterintuitive
convention baked into the experiment outputs: `"ips"` holds performance on the
**external temporal benchmark**, while `"test"` holds performance on the
**internal CV test split** (the optimistically biased held-out 20%). Estimation
bias / optimism is therefore `test − ips` (internal − benchmark), and benchmark
performance is the `"ips"` value. Verified empirically against the reported
results (e.g. SPXY benchmark Spearman ≈ 0.63 = `ips`; optimism ≈ +0.22 = test − ips).
"""
function layer1_to_long(df::DataFrame)
    id_cols  = [:dataset, :algorithm, :ratio, :model, :rep, :fold]
    ips_src  = [:r2_ips,  :rmse_ips,  :mae_ips,  :bias_ips]
    test_src = [:r2_test, :rmse_test, :mae_test, :bias_test]

    df2 = rename(df, :splitter => :algorithm)

    ips_df = select(df2, id_cols..., ips_src...)
    rename!(ips_df, Dict(zip(ips_src, LAYER1_METRICS)))
    ips_df[!, :set] .= "ips"

    test_df = select(df2, id_cols..., test_src...)
    rename!(test_df, Dict(zip(test_src, LAYER1_METRICS)))
    test_df[!, :set] .= "test"

    vcat(ips_df, test_df)
end
