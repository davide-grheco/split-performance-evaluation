using Clustering
using Distances
using DataSplits
using MLJ
using MLUtils
using Random
using .Config: SplittingConfig, ModelsConfig

const _LGBMRegressor = MLJ.@load LGBMRegressor pkg = LightGBM verbosity = 0
const _RFRegressor = MLJ.@load RandomForestRegressor pkg = DecisionTree verbosity = 0

# ---------------------------------------------------------------------------
# Butina wrapper — bridges the internal butina_split() to DataSplits.SplitStrategy
# ---------------------------------------------------------------------------

struct ButinaSplitStrategy <: DataSplits.SplitStrategy
    frac::Float64
    cutoff::Float64
end

function DataSplits.split(X, ::Any, strategy::ButinaSplitStrategy;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    DataSplits.split(X, strategy; rng=rng)
end

function DataSplits.split(X, strategy::ButinaSplitStrategy;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    butina_split(X; frac=strategy.frac, cutoff=strategy.cutoff, rng=rng)
end

# ---------------------------------------------------------------------------
# Clustering-based split wrapper
# ---------------------------------------------------------------------------
# Self-contained: runs clustering inside split(), so no pre-computation step.

struct ClusteringBasedSplit <: DataSplits.SplitStrategy
    frac::Float64
    k::Int
    algorithm::Symbol   # :kmeans | :kmedoids | :hac
    mode::Symbol        # :stratified | :shuffle
end

function _clustering_based_assignments(X::AbstractMatrix, alg::Symbol, k::Int; rng::AbstractRNG)
    if alg == :kmeans
        data = Float64.(X)   # kmeans needs numeric features×samples matrix
        return Clustering.kmeans(data, k; init=:kmpp, maxiter=300, display=:none, rng=rng)
    elseif alg == :kmedoids
        D = Distances.pairwise(Distances.Jaccard(), X)
        return Clustering.kmedoids(D, k; display=:none)
    elseif alg == :hac
        D = Distances.pairwise(Distances.Jaccard(), X)
        tree = Clustering.hclust(D; linkage=:average)
        asgn = Clustering.cutree(tree; k=k)
        cnts = [count(==(i), asgn) for i in 1:maximum(asgn)]
        return ButinaResult(asgn, cnts, NaN)   # ButinaResult implements ClusteringResult
    else
        error("Unknown clustering algorithm: $alg. Use :kmeans, :kmedoids, or :hac.")
    end
end

function DataSplits.split(X, ::Any, strategy::ClusteringBasedSplit;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    DataSplits.split(X, strategy; rng=rng)
end

function DataSplits.split(X, strategy::ClusteringBasedSplit;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    cr = _clustering_based_assignments(X, strategy.algorithm, strategy.k; rng=rng)
    inner = if strategy.mode == :shuffle
        DataSplits.ClusterShuffleSplit(cr, strategy.frac)
    else
        DataSplits.ClusterStratifiedSplit(cr, :proportional; frac=strategy.frac)
    end
    return DataSplits.split(X, inner; rng=rng)
end

# ---------------------------------------------------------------------------
# Fast maximum dissimilarity — uses the optimized optisim() kernel with a
# precomputed Jaccard distance matrix and incremental min_dist bookkeeping,
# bypassing the DataSplits v0.1 lazy implementation.
# ---------------------------------------------------------------------------

struct FastMaximumDissimilaritySplitStrategy <: DataSplits.SplitStrategy
    frac::Float64
    distance_cutoff::Float64
end

function DataSplits.split(X, ::Any, strategy::FastMaximumDissimilaritySplitStrategy;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    DataSplits.split(X, strategy; rng=rng)
end

function DataSplits.split(X, strategy::FastMaximumDissimilaritySplitStrategy;
    rng::AbstractRNG=Random.GLOBAL_RNG)
    N = MLUtils.numobs(X)
    n_train = round(Int, strategy.frac * N)
    D = Distances.pairwise(Distances.Jaccard(), X)
    selected = optisim(D, n_train, 0, strategy.distance_cutoff; rng=rng)
    train_pos = collect(selected)
    test_pos = setdiff(1:N, train_pos)
    return (train=train_pos, test=test_pos)
end

# ---------------------------------------------------------------------------
# Splitter registry
# ---------------------------------------------------------------------------
# Each entry is a factory: (frac::Float64, cfg::SplittingConfig) -> SplitStrategy
# Adding a new splitter = one Dict entry; no other files need changing.

const SPLITTER_REGISTRY = Dict{String,Function}(
    "random" => (frac, _cfg) -> DataSplits.RandomSplit(frac),
    "kennardstone" => (frac, _cfg) -> DataSplits.LazyKennardStoneSplit(frac, Jaccard()),
    "mdks" => (frac, _cfg) -> DataSplits.MDKSSplit(frac),
    "butina" => (frac, _cfg) -> ButinaSplitStrategy(frac, 0.2),
    "spxy-jaccard" => (frac, _cfg) -> DataSplits.SPXYSplit(frac, metric_X=Distances.Jaccard(), metric_y=Distances.Euclidean()),
    "spxy-euclidean" => (frac, _cfg) -> DataSplits.SPXYSplit(frac, metric_X=Distances.Euclidean(), metric_y=Distances.Euclidean()),
    "optisim" => (frac, _cfg) -> DataSplits.LazyOptiSimSplit(frac, max_subsample_size=50, distance_cutoff=0.35, metric=Distances.Jaccard()),
    "maximum_dissimilarity" => (frac, _cfg) -> FastMaximumDissimilaritySplitStrategy(frac, 0.35),
    "minimum_dissimilarity" => (frac, _cfg) -> DataSplits.LazyMinimumDissimilaritySplit(frac, distance_cutoff=0.35, metric=Distances.Jaccard()),
    "morais" => (frac, _cfg) -> DataSplits.MoraisLimaMartinSplit(frac, swap_frac=0.1, metric=Distances.Jaccard()),
    "kmeans_stratified"   => (frac, _) -> ClusteringBasedSplit(frac, 10, :kmeans,   :stratified),
    "kmeans_shuffle"      => (frac, _) -> ClusteringBasedSplit(frac, 10, :kmeans,   :shuffle),
    "kmedoids_stratified" => (frac, _) -> ClusteringBasedSplit(frac, 10, :kmedoids, :stratified),
    "kmedoids_shuffle"    => (frac, _) -> ClusteringBasedSplit(frac, 10, :kmedoids, :shuffle),
    "hac_stratified"      => (frac, _) -> ClusteringBasedSplit(frac, 10, :hac,      :stratified),
    "hac_shuffle"         => (frac, _) -> ClusteringBasedSplit(frac, 10, :hac,      :shuffle),
)

"""
    make_splitter(name, frac, cfg) -> SplitStrategy

Construct the splitter identified by `name` using the given train fraction
and any additional parameters from `cfg`.
"""
function make_splitter(name::String, frac::Float64, cfg::SplittingConfig)
    factory = get(SPLITTER_REGISTRY, name, nothing)
    factory === nothing && error("Unknown splitter: \"$name\". Available: $(join(sort(collect(keys(SPLITTER_REGISTRY))), ", "))")
    return factory(frac, cfg)
end

make_splitter(name::String, cfg::SplittingConfig) = make_splitter(name, first(cfg.ratios), cfg)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

"""
    make_model(name, cfg; seed) -> MLJ model instance

Construct the MLJ model identified by `name` using parameters from `cfg`.
Add new models by inserting an `elseif` branch — no other files need changing.
"""
function make_model(name::String, cfg::ModelsConfig; seed::Integer=42)
    if name == "lightgbm"
        return _LGBMRegressor(
            num_iterations=cfg.lightgbm.n_estimators,
            learning_rate=cfg.lightgbm.learning_rate,
            feature_fraction=0.8,
            feature_fraction_seed=Int(seed),
            num_threads=1,   # Julia provides parallelism; disable LightGBM OpenMP threads
            verbosity=-1,
        )
    elseif name == "random_forest"
        return _RFRegressor(rng=MersenneTwister(seed))
    else
        error("Unknown model: \"$name\". Add it to make_model.")
    end
end

# MODEL_REGISTRY placeholder: name -> model constructor
const MODEL_REGISTRY = Dict{String,Function}()

# AD_REGISTRY placeholder: name -> applicability domain constructor
const AD_REGISTRY = Dict{String,Function}()
