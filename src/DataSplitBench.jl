module DataSplitBench

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Data loading
export merck_dataloader, merck_train_dataloader, merck_split_dataloader, load_outer_cv_subset
# Splitting
export apply_split
# Precomputation / I/O
export load_split, load_cv, save_cv, load_method_split
# Registry
export make_splitter, make_model, SPLITTER_REGISTRY, MODEL_REGISTRY, AD_REGISTRY
# Model selection
export tpe_search, select_champion
# Applicability domain
export fit_ad, ad_score, in_domain, KNNDistanceAD
# Metrics
export nn_distance, evaluate_split_metrics
# Experiment loading (Arrow-based)
export load_layer1_summary, load_split_metrics, layer1_to_long,
    load_ad_predictions, load_ad_champion, load_champion_meta,
    load_ad_summary, load_champion_summary,
    LAYER1_METRICS, LAYER1_HIGHER_BETTER
# Analysis utilities
export dataset_names, metrics_results, merge_repeated_run_results, mean_scores_by_set,
    benchmark_summary, benchmark_gap, invert_to_minimize, is_higher_better, HIGHER_BETTER_DEFAULT,
    perdataset_means, average_ranks_across_datasets, splitter_family, short_code,
    collapse_dataset_algorithm, per_dataset_spread_df, quadrant_summary_df,
    threshold_counts_df, find_homogeneous_rank_spans, layout_group_bar_rows, pairwise_to_matrix,
    compute_fold_metrics, compute_ad_coverage, nemenyi_from_df,
    compute_fold_selection_quality, aggregate_selection_by_dataset
# Statistical tests
export FriedmanTest, iman_davenport, NemenyiResult, nemenyi, nemenyi_after_iman
# Multicriteria decision analysis (TOPSIS / MOORA)
export MCDAResult, topsis_rank, moora_rank, mcda_rank
# Plotting
export cd_diagram, nemenyi_heatmap, plot_per_dataset_spread, DEFAULT_COLOR_SCHEME

# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------
include("Config.jl")
include("Butina.jl")
include("Splitters.jl")
include("Registry.jl")
include("Metrics.jl")
include("DataLoader.jl")
include("Chemistry.jl")
include("Model.jl")
include("Utils.jl")
include("Precomputation.jl")
include("Friedman.jl")
include("Nemenyi.jl")
include("MCDA.jl")
include("AnalysisUtils.jl")
include("Plotting.jl")
include("ExperimentLoader.jl")
include("ApplicabilityDomain.jl")
include("ModelSelection.jl")
include("Data/Nci-60.jl")

end
