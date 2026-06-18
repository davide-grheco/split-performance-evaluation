module Config

using TOML

# ---------------------------------------------------------------------------
# Sub-config structs
# ---------------------------------------------------------------------------

struct DataConfig
    root::String
    cv_root::String
    datasets::Vector{String}
    subsample_n::Union{Int,Nothing}
    subsample_seed_offset::Union{Int,Nothing}
end

struct SplittingConfig
    methods::Vector{String}
    ratios::Vector{Float64}   # train fractions to evaluate
    repeats::Int
    k_folds::Int
end

struct LightGBMConfig
    n_estimators::Int
    learning_rate::Float64
end

struct ModelsConfig
    methods::Vector{String}
    n_runs::Int                # independent retraining runs per split
    tpe_budget::Int            # number of TPE evaluations per hyperopt run
    lightgbm::LightGBMConfig
end

struct OutputConfig
    root::String
end

struct ADConfig
    method::String    # "knn_distance" (the only supported method for now)
    k::Int            # number of neighbours for KNNDistanceAD
    z::Float64        # threshold = mean(loo_knn) + z * std(loo_knn)
end

struct Layer2Config
    model_candidates::Vector{String}   # model names passed to select_champion()
    tpe_budget::Int                    # TPE evaluations per model candidate
    tpe_subsample_n::Union{Int,Nothing}  # cap training rows used during TPE trials (nothing = no cap)
end

struct ExperimentConfig
    name::String
    description::String
    base_seed::Int
    data::DataConfig
    splitting::SplittingConfig
    models::ModelsConfig
    ad::ADConfig
    output::OutputConfig
    layer2::Union{Layer2Config,Nothing}   # nothing → Layer 2 workflow disabled
end

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

function load_config(path::String)::ExperimentConfig
    raw = TOML.parsefile(path)

    exp = raw["experiment"]
    name = exp["name"]
    description = get(exp, "description", "")
    base_seed = get(exp, "base_seed", 42)

    d = raw["data"]
    data = DataConfig(
        d["root"],
        d["cv_root"],
        d["datasets"],
        haskey(d, "subsample_n") ? d["subsample_n"] : nothing,
        haskey(d, "subsample_seed_offset") ? d["subsample_seed_offset"] : nothing,
    )

    s = raw["splitting"]
    # Accept either `ratios` (list) or legacy `frac` (scalar)
    ratios = if haskey(s, "ratios")
        Float64.(s["ratios"])
    else
        [Float64(get(s, "frac", 0.8))]
    end
    splitting = SplittingConfig(
        s["methods"],
        ratios,
        get(s, "repeats", 5),
        get(s, "k_folds", 5),
    )

    m = raw["models"]
    lgbm_raw = get(m, "lightgbm", Dict())
    lgbm = LightGBMConfig(
        get(lgbm_raw, "n_estimators", 500),
        get(lgbm_raw, "learning_rate", 0.05),
    )
    models = ModelsConfig(
        m["methods"],
        get(m, "n_runs", 10),
        get(m, "tpe_budget", 50),
        lgbm,
    )

    ad_raw = get(raw, "applicability_domain", Dict())
    ad = ADConfig(
        get(ad_raw, "method", "knn_distance"),
        get(ad_raw, "k",      5),
        Float64(get(ad_raw, "z", 1.5)),
    )

    l2_raw = get(raw, "layer2", nothing)
    layer2 = if l2_raw !== nothing
        Layer2Config(
            get(l2_raw, "model_candidates", ["lightgbm"]),
            get(l2_raw, "tpe_budget", get(m, "tpe_budget", 50)),
            haskey(l2_raw, "tpe_subsample_n") ? Int(l2_raw["tpe_subsample_n"]) : nothing,
        )
    else
        nothing
    end

    o = raw["output"]
    output = OutputConfig(o["root"])

    return ExperimentConfig(name, description, base_seed, data, splitting, models, ad, output, layer2)
end

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

experiment_root(cfg::ExperimentConfig) = joinpath(cfg.output.root, cfg.name)
splits_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "splits")
predictions_shards_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "predictions", "shards")
predictions_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "predictions")

# Kept for backward compatibility during transition
metrics_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "metrics")
model_results_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "model_results")

# Layer 1 summary
layer1_summary_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "layer1_summary")

# Distribution-shift metrics (per split)
split_metrics_shards_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "split_metrics", "shards")
split_metrics_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "split_metrics")

# Layer 2 champion predictions
champion_shards_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "champion_predictions", "shards")
champion_predictions_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "champion_predictions")

# Layer 3 on champion predictions
layer3_champion_shards_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "ad_champion", "shards")
layer3_champion_root(cfg::ExperimentConfig) = joinpath(experiment_root(cfg), "ad_champion")


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

"""
    load_experiment_roots(; data_default, split_default) -> NamedTuple

Load the experiment config from `ENV["EXPERIMENT_CONFIG"]` (if set) and
return a named tuple `(cfg, data_root, split_root)`.

When no config file is present, `data_root` falls back to
`ENV["DATA_ROOT"]` (or `data_default`) and `split_root` falls back to
`ENV["SPLIT_ROOT"]` (or `split_default`).
"""
function load_experiment_roots(;
    data_default::String="Data/merck/",
    split_default::String="Splits/merck/")
    cfg_path = get(ENV, "EXPERIMENT_CONFIG", "")
    cfg = isempty(cfg_path) ? nothing : load_config(cfg_path)
    data_root = cfg !== nothing ? cfg.data.root : get(ENV, "DATA_ROOT", data_default)
    split_root = cfg !== nothing ? cfg.data.cv_root : get(ENV, "SPLIT_ROOT", split_default)
    return (cfg=cfg, data_root=data_root, split_root=split_root)
end

end  # module Config
