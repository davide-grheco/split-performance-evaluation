# ModelSelection.jl
# Hyperparameter optimisation via Tree-structured Parzen Estimator (TPE)
# with successive halving for champion selection.
#
# To add a new model:
#   1. Define a struct   MyModelSpec <: ModelSpec
#   2. Add it to MODEL_SPECS
#   3. Define _tpe_space(::MyModelSpec)                -> Dict{Symbol, <:HP.*}
#   4. Define _build_model(::MyModelSpec, params, seed) -> MLJ model instance
#   5. Define _params_to_dict(::MyModelSpec, params)   -> Dict{String,Any}

using MLJ
using Random
using TreeParzen
using Distances
using LinearAlgebra
using Statistics
import LIBSVM

# @load at module level so the type is resolved once.
const _LGBMRegressor = MLJ.@load LGBMRegressor pkg = LightGBM verbosity = 0
const _RFRegressor   = MLJ.@load RandomForestRegressor  pkg = DecisionTree     verbosity = 0
const _ElNetRegressor = MLJ.@load ElasticNetRegressor   pkg = MLJLinearModels  verbosity = 0

# ---------------------------------------------------------------------------
# ThreadedEpsilonSVR — thin MLJ wrapper around LIBSVM.svmtrain that exposes
# the `nt` (thread count) parameter, which MLJLIBSVMInterface hard-codes to 1.
# ---------------------------------------------------------------------------

mutable struct ThreadedEpsilonSVR <: MLJ.Deterministic
    kernel::LIBSVM.Kernel.KERNEL
    gamma::Float64
    epsilon::Float64
    cost::Float64
    cachesize::Float64
    degree::Int
    coef0::Float64
    tolerance::Float64
    shrinking::Bool
    nt::Int
end

function ThreadedEpsilonSVR(;
    kernel=LIBSVM.Kernel.RadialBasis, gamma=0.0, epsilon=0.1,
    cost=1.0, cachesize=200.0, degree=3, coef0=0.0,
    tolerance=0.001, shrinking=true, nt=1,
)
    ThreadedEpsilonSVR(kernel, gamma, epsilon, cost, cachesize,
                       degree, coef0, tolerance, shrinking, nt)
end

function MLJ.fit(m::ThreadedEpsilonSVR, ::Int, X, y)
    Xmat = MLJ.matrix(X)'
    # Replicate MLJLIBSVMInterface gamma auto-scaling: 0.0 → 1/(var·nfeatures)
    γ = m.gamma == 0.0 ? 1.0 / (var(Xmat) * size(Xmat, 1)) : Float64(m.gamma)
    fitresult = LIBSVM.svmtrain(Xmat, collect(Float64, y);
        svmtype=LIBSVM.EpsilonSVR,
        kernel=m.kernel, gamma=γ, epsilon=m.epsilon, cost=m.cost,
        cachesize=m.cachesize, degree=Int32(m.degree), coef0=m.coef0,
        tolerance=m.tolerance, shrinking=m.shrinking,
        verbose=false, nt=m.nt,
    )
    return fitresult, nothing, (gamma=γ,)
end

function MLJ.predict(m::ThreadedEpsilonSVR, fitresult, Xnew)
    p, _ = LIBSVM.svmpredict(fitresult, MLJ.matrix(Xnew)'; nt=m.nt)
    return p
end

# ---------------------------------------------------------------------------
# MatrixKNNRegressor — custom brute-force KNN that avoids StaticArrays.
#
# NearestNeighborModels.KNNRegressor routes data through NearestNeighbors.BruteTree,
# which internally calls copy_svec(T, data, Val(D)) → Vector{SVector{D, T}}.
# For high-dimensional fingerprints (D ≈ 8000+) StaticArrays tries to specialise
# generated functions for that D, exhausting Julia's GC-handle limit.
#
# This implementation stores training data as a plain Matrix and queries with
# Distances.colwise, which operates on ordinary arrays and has no such limit.
# ---------------------------------------------------------------------------

mutable struct MatrixKNNRegressor <: MLJ.Deterministic
    K::Int
    weights_idx::Int   # 0 = Uniform, 1 = InverseDistance
    metric_idx::Int    # 0 = Euclidean, 1 = Cityblock
end

const _MKNN_METRICS = [Euclidean(), Cityblock()]

function MLJ.fit(m::MatrixKNNRegressor, ::Int, X, y)
    Xmat = MLJ.matrix(X)'   # features × samples
    return (Xmat=Xmat, y=collect(Float64, y)), nothing, NamedTuple()
end

function MLJ.predict(m::MatrixKNNRegressor, fitresult, Xnew)
    Xnew_mat = MLJ.matrix(Xnew)'
    Xmat, ytrain = fitresult.Xmat, fitresult.y
    metric = _MKNN_METRICS[m.metric_idx+1]
    K       = min(m.K, length(ytrain))
    n_test  = size(Xnew_mat, 2)
    n_train = length(ytrain)
    preds   = Vector{Float64}(undef, n_test)
    # One distance buffer per thread — avoids allocating an n_test×n_train matrix
    # (≈1.6 GB at LOGD/3A4 scale) and parallelises over test points instead.
    bufs = [Vector{Float64}(undef, n_train) for _ in 1:Base.Threads.maxthreadid()]
    Threads.@threads for i in 1:n_test
        d = bufs[Threads.threadid()]
        Distances.colwise!(metric, d, @view(Xnew_mat[:, i]), Xmat)
        idxs = partialsortperm(d, 1:K)
        if m.weights_idx == 0   # Uniform
            preds[i] = mean(@view ytrain[idxs])
        else                    # InverseDistance
            w = 1.0 ./ @view(d[idxs])
            any(isinf, w) && (w = Float64.(isinf.(w)))  # exact match: unit weight
            preds[i] = dot(w, @view(ytrain[idxs])) / sum(w)
        end
    end
    return preds
end

export tpe_search, select_champion, CandidateResult, retrain_candidate

# ---------------------------------------------------------------------------
# Result type returned by select_champion
# ---------------------------------------------------------------------------

"""
    CandidateResult

Holds the outcome of TPE search for one model candidate after champion selection.

Fields:
- `name`             — model identifier string
- `params`           — best hyperparameters found (human-readable Dict)
- `cv_rms`           — best inner-CV RMS achieved (lower is better)
- `raw_params`       — raw TreeParzen parameter dict; pass to `retrain_candidate`
- `is_champion`      — true for the single winner
- `round_eliminated` — successive halving round in which this model was
                        eliminated (1 or 2); 0 means it survived all rounds
                        (i.e., it is the champion)
"""
struct CandidateResult
    name::String
    params::Dict{String,Any}
    cv_rms::Float64
    raw_params::Dict{Symbol,Any}  # raw TreeParzen params; used by retrain_candidate
    is_champion::Bool
    round_eliminated::Int
end

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

abstract type ModelSpec end

struct LightGBMSpec <: ModelSpec end
struct RandomForestSpec <: ModelSpec end
struct KNNSpec <: ModelSpec end
struct SVRSpec <: ModelSpec end
struct ElasticNetSpec <: ModelSpec end

const MODEL_SPECS = Dict{Symbol,ModelSpec}(
    :lightgbm => LightGBMSpec(),
    :random_forest => RandomForestSpec(),
    :knn => KNNSpec(),
    :svr => SVRSpec(),
    :elasticnet => ElasticNetSpec(),
)

function _resolve_spec(name::Union{String,Symbol})
    sym = Symbol(name)
    spec = get(MODEL_SPECS, sym, nothing)
    spec === nothing && error(
        "Unknown model: $(repr(name)). " *
        "Available: $(join(sort(string.(keys(MODEL_SPECS))), ", "))"
    )
    return spec
end

# ---------------------------------------------------------------------------
# Type-dispatch pre-warming
# ---------------------------------------------------------------------------

# 30 rows: enough for 2-fold CV with KNN(K=5) (≥6 training rows per fold).
const _PREWARM_N = 30

_default_prewarm_params(::LightGBMSpec) = Dict{Symbol,Any}(:num_iterations => 100.0, :learning_rate => 0.1, :num_leaves => 31.0, :feature_fraction => 1.0, :min_data_in_leaf => 20.0)
_default_prewarm_params(::RandomForestSpec) = Dict{Symbol,Any}(:n_trees => 50.0, :max_depth => -1.0, :min_samples_leaf => 1.0, :min_samples_split => 2.0)
_default_prewarm_params(::KNNSpec) = Dict{Symbol,Any}(:K => 5.0, :weights_idx => 0.0, :metric_idx => 0.0)
_default_prewarm_params(::SVRSpec) = Dict{Symbol,Any}(:cost => 1.0, :gamma => 0.1, :epsilon => 0.1)
_default_prewarm_params(::ElasticNetSpec) = Dict{Symbol,Any}(:lambda => 1.0, :gamma => 0.5)

# Run one 2-fold CV evaluation per model type on a tiny slice of the real data.
# This triggers all MLJ type-dispatch compilation upfront, before the heap is
# loaded with trial objects, avoiding mid-search "out of gc handles" crashes.
function _prewarm_dispatch!(specs, X_train, y_train)
    n_warm = min(_PREWARM_N, MLJ.nrows(X_train))
    X_tiny = selectrows(X_train, 1:n_warm)
    y_tiny = y_train[1:n_warm]
    resampling = CV(nfolds=2, rng=0)
    @info "Pre-warming MLJ type dispatch" n_models = length(specs)
    for spec in specs
        try
            model = _build_model(spec, _default_prewarm_params(spec), 0)
            MLJ.evaluate(model, X_tiny, y_tiny;
                resampling=resampling, measure=rms, verbosity=0)
        catch err
            @warn "Pre-warm failed (non-fatal)" model_spec = typeof(spec) exception = err
        end
    end
    GC.gc(true)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    select_champion(model_names, X_train, y_train; budget, n_inner_folds, base_seed)
        -> Vector{CandidateResult}

Run successive halving TPE across all `model_names` and return one
`CandidateResult` per candidate, ordered as given.

Successive halving schedule (derived from `budget` and `n_inner_folds`):

  Round 1 — coarse screen:   all models, budget_r1=5,          n_folds=2,          keep top 2
  Round 2 — refined search:  top 2,      budget_r2=(B-5)÷2,   n_folds=n_inner_folds, keep top 1
  Round 3 — final polish:    champion,   budget_r3=B-5-budget_r2, n_folds=n_inner_folds

TPE trial history accumulates across rounds so the champion benefits from
all `budget` total observations. Eliminated models keep their best config
from whichever round they exited.

After selection, **all** candidates are retrained on the full training set so
that callers can record external-set predictions for every model (needed for
oracle regret analysis).

Models are processed **sequentially** so each model has exclusive access to
all `Threads.nthreads()` Julia threads during its own TPE loop.

When `tpe_subsample_n` is set, TPE trials use at most that many randomly
sampled rows from `X_train`/`y_train`.  Final retraining via
`retrain_candidate` always uses the caller's full training set.
"""
function select_champion(model_names, X_train, y_train;
    budget::Int=50, n_inner_folds::Int=5, base_seed::Integer=42,
    tpe_subsample_n::Union{Int,Nothing}=nothing)
    n = length(model_names)

    # Derive round schedule from budget
    r1_budget = 5
    remaining = budget - r1_budget
    r2_budget = remaining ÷ 2
    r3_budget = remaining - r2_budget
    rounds = (
        (budget=r1_budget, n_folds=2, keep=2),
        (budget=r2_budget, n_folds=n_inner_folds, keep=1),
        (budget=r3_budget, n_folds=n_inner_folds, keep=1),
    )

    # TPE config: n_random based on champion's total budget (all rounds summed)
    total_champion_budget = r1_budget + r2_budget + r3_budget
    n_random = clamp(total_champion_budget ÷ 3, 5, 20)

    # Subsample training data for TPE trials when the dataset is large.
    # retrain_candidate always uses the full (X_train, y_train) passed by the caller.
    n_full = MLJ.nrows(X_train)
    if tpe_subsample_n !== nothing && n_full > tpe_subsample_n
        rng_sub = Random.MersenneTwister(base_seed + 7919)
        sub_idx = sort(Random.randperm(rng_sub, n_full)[1:tpe_subsample_n])
        X_tpe = selectrows(X_train, sub_idx)
        y_tpe = y_train[sub_idx]
        @info "TPE subsample" n_full tpe_subsample_n
    else
        X_tpe = X_train
        y_tpe = y_train
    end

    # Per-model state
    specs = [_resolve_spec(name) for name in model_names]
    spaces = [_tpe_space(spec) for spec in specs]
    configs = [TreeParzen.Config(0.25, 25, 24, n_random, 1.0) for _ in 1:n]
    trials = [TreeParzen.Trials.Trial[] for _ in 1:n]
    best_raws = Vector{Any}(undef, n)
    cv_rmsv = fill(Inf, n)
    round_elim = fill(0, n)

    active = collect(1:n)  # indices of models still in competition

    # Front-load all JIT compilation before the heap fills with trial objects.
    _prewarm_dispatch!(specs, X_tpe, y_tpe)

    for (round_idx, (budget_incr, n_folds, keep)) in enumerate(rounds)
        isempty(active) && break
        resampling = CV(nfolds=n_folds, rng=base_seed)

        for i in active
            @info "  SH round $round_idx" model = model_names[i] budget_incr n_folds
            best_raws[i], cv_rmsv[i] = _tpe_run_round!(
                trials[i], configs[i], spaces[i], specs[i],
                X_tpe, y_tpe, budget_incr, resampling, base_seed,
            )
            @info "  Result" model = model_names[i] cv_rms = cv_rmsv[i]
        end

        # Eliminate weakest models; keep at most `keep` survivors.
        # Free trial history for eliminated models immediately.
        keep_actual = min(keep, length(active))
        if keep_actual < length(active)
            sorted = sort(active; by=i -> cv_rmsv[i])
            for i in sorted[keep_actual+1:end]
                round_elim[i] = round_idx
                trials[i] = TreeParzen.Trials.Trial[]  # free accumulated trial objects
            end
            active = sorted[1:keep_actual]
        end
        GC.gc(true)
    end

    champion_i = active[1]

    # Build results with raw_params so callers can retrain one model at a time
    # via retrain_candidate, keeping only one fitted machine live at once.
    results = Vector{CandidateResult}(undef, n)
    for i in 1:n
        results[i] = CandidateResult(
            model_names[i],
            _params_to_dict(specs[i], best_raws[i]),
            cv_rmsv[i],
            best_raws[i],
            i == champion_i,
            round_elim[i],
        )
    end

    return results
end

"""
    tpe_search(name, X_train, y_train; budget=50, n_inner_folds=5, seed=42)
        -> (best_params::Dict{String,Any}, best_cv_rms::Float64, mach)

Run a flat TPE hyperparameter search for a single model (no successive halving).
Used by `run_hyperopt.jl` for single-model jobs.

Returns:
- `best_params`  — human-readable `Dict{String,Any}` of best hyperparameters
- `best_cv_rms`  — inner-CV RMS of the best trial
- `mach`         — MLJ machine retrained on the full training set
"""
function tpe_search(name::Union{String,Symbol}, X_train, y_train;
    budget::Int=50, n_inner_folds::Int=5, seed::Integer=42)
    spec = _resolve_spec(name)
    space = _tpe_space(spec)
    n_random = clamp(budget ÷ 3, 5, 20)
    config = TreeParzen.Config(0.25, 25, 24, n_random, 1.0)
    trials = TreeParzen.Trials.Trial[]
    resampling = CV(nfolds=n_inner_folds, rng=seed)

    best_raw, cv_rms = _tpe_run_round!(trials, config, space, spec,
        X_train, y_train, budget, resampling, seed)
    mach = _fit_best(spec, best_raw, X_train, y_train, seed)
    return _params_to_dict(spec, best_raw), cv_rms, mach
end

# ---------------------------------------------------------------------------
# Internal TPE primitives
# ---------------------------------------------------------------------------

# Run `budget_incr` more TPE trials, mutating `trials` in place.
# Returns (best_raw_params, best_cv_rms) from the full accumulated trial history.
function _tpe_run_round!(trials::Vector, config, space, spec::ModelSpec,
    X_train, y_train, budget_incr::Int, resampling, seed::Integer)
    for _ in 1:budget_incr
        trial = TreeParzen.ask(space, trials, config)
        loss = _cv_rms(spec, trial.hyperparams, X_train, y_train, resampling, seed)
        TreeParzen.tell!(trials, trial, loss)
        GC.gc()
    end
    best_raw = TreeParzen.provide_recommendation(trials)
    best_cv_rms = minimum(t.loss for t in trials)
    return best_raw, best_cv_rms
end

# Build and fit a model from raw TreeParzen hyperparameter dict on the full training set.
function _fit_best(spec::ModelSpec, best_raw::Dict, X_train, y_train, seed::Integer)
    model = _build_model(spec, best_raw, seed)
    mach = machine(model, X_train, y_train)
    fit!(mach; verbosity=0)
    return mach
end

"""
    retrain_candidate(result, X_train, y_train, base_seed) -> machine

Retrain the model described by `result` on `(X_train, y_train)` and return
the fitted MLJ machine.  Call this once per candidate so that only one machine
is live at a time, keeping peak memory proportional to the largest single model
rather than the sum of all candidates.

`base_seed` must match the value passed to `select_champion`.
"""
function retrain_candidate(result::CandidateResult, X_train, y_train, base_seed::Integer)
    seed = Int(abs(hash((base_seed, result.name))) % typemax(Int32))
    spec = _resolve_spec(result.name)
    return _fit_best(spec, result.raw_params, X_train, y_train, seed)
end

# RandomForest uses a shared MersenneTwister internally; parallel fold tasks would
# concurrently resize its internal buffer → ConcurrencyViolationError (Julia ≥1.12).
# Use sequential folds instead and let DecisionTree.build_forest consume all threads.
_fold_acceleration(::ModelSpec)        = CPUThreads()
_fold_acceleration(::RandomForestSpec) = CPU1()

# Run inner CV and return mean RMS for one hyperparameter sample.
function _cv_rms(spec::ModelSpec, params::Dict, X_train, y_train, resampling, seed::Integer)
    # Divide Julia threads evenly across folds so total CPU use stays at nthreads().
    lgbm_threads = max(1, Threads.nthreads() ÷ resampling.nfolds)
    model = _build_model(spec, params, seed; nthreads=lgbm_threads)
    result = MLJ.evaluate(model, X_train, y_train;
        resampling=resampling, measure=rms, verbosity=0,
        acceleration=_fold_acceleration(spec))
    return result.measurement[1]
end

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

"""
TPE search space for LightGBM:
  num_iterations    ~ QuantUniform(100, 500, step=100)   [100, 200, 300, 400, 500]
  learning_rate     ~ LogUniform(log(0.005), log(0.3))   continuous
  num_leaves        ~ QuantUniform(15, 255, step=1)       integer-valued
  feature_fraction  ~ Uniform(0.5, 1.0)                  continuous
  min_data_in_leaf  ~ QuantUniform(5, 100, step=5)        integer-valued; default 20
"""
function _tpe_space(::LightGBMSpec)
    Dict{Symbol,Any}(
        :num_iterations => HP.QuantUniform(:num_iterations, 100.0, 500.0, 100.0),
        :learning_rate => HP.LogUniform(:learning_rate, log(0.005), log(0.3)),
        :num_leaves => HP.QuantUniform(:num_leaves, 15.0, 255.0, 1.0),
        :feature_fraction => HP.Uniform(:feature_fraction, 0.5, 1.0),
        :min_data_in_leaf => HP.QuantUniform(:min_data_in_leaf, 5.0, 100.0, 5.0),
    )
end

function _build_model(::LightGBMSpec, params::Dict, seed::Integer;
                      nthreads::Int=Threads.nthreads())
    _LGBMRegressor(
        num_iterations=Int(round(params[:num_iterations])),
        learning_rate=Float64(params[:learning_rate]),
        num_leaves=Int(round(params[:num_leaves])),
        feature_fraction=Float64(params[:feature_fraction]),
        # Default 20 matches LightGBM's built-in default; old results remain valid.
        min_data_in_leaf=Int(round(get(params, :min_data_in_leaf, 20.0))),
        feature_fraction_seed=Int(seed),
        num_threads=nthreads,
        verbosity=-1,
    )
end

function _params_to_dict(::LightGBMSpec, params::Dict)
    Dict{String,Any}(
        "num_iterations" => Int(round(params[:num_iterations])),
        "learning_rate" => Float64(params[:learning_rate]),
        "num_leaves" => Int(round(params[:num_leaves])),
        "feature_fraction" => Float64(params[:feature_fraction]),
        "min_data_in_leaf" => Int(round(get(params, :min_data_in_leaf, 20.0))),
    )
end

# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

"""
TPE search space for Random Forest:
  n_trees            ~ QuantUniform(50, 500, step=50)    [50, 100, …, 500]
  max_depth          ~ Choice([-1, 5, 10, 20])           discrete
  min_samples_leaf   ~ QuantUniform(1, 20, step=1)       integer-valued
  min_samples_split  ~ QuantUniform(2, 20, step=1)       integer-valued; default 2
"""
function _tpe_space(::RandomForestSpec)
    Dict{Symbol,Any}(
        :n_trees => HP.QuantUniform(:n_trees, 50.0, 500.0, 50.0),
        :max_depth => HP.Choice(:max_depth, [-1.0, 5.0, 10.0, 20.0]),
        :min_samples_leaf => HP.QuantUniform(:min_samples_leaf, 1.0, 20.0, 1.0),
        :min_samples_split => HP.QuantUniform(:min_samples_split, 2.0, 20.0, 1.0),
    )
end

function _build_model(::RandomForestSpec, params::Dict, seed::Integer; nthreads::Int=Threads.nthreads())
    _RFRegressor(
        n_trees=Int(round(params[:n_trees])),
        max_depth=Int(round(params[:max_depth])),
        min_samples_leaf=Int(round(params[:min_samples_leaf])),
        # Default 2 matches DecisionTree.jl's built-in default; old results remain valid.
        min_samples_split=Int(round(get(params, :min_samples_split, 2.0))),
        rng=MersenneTwister(seed),
    )
end

function _params_to_dict(::RandomForestSpec, params::Dict)
    Dict{String,Any}(
        "n_trees" => Int(round(params[:n_trees])),
        "max_depth" => Int(round(params[:max_depth])),
        "min_samples_leaf" => Int(round(params[:min_samples_leaf])),
        "min_samples_split" => Int(round(get(params, :min_samples_split, 2.0))),
    )
end

# ---------------------------------------------------------------------------
# k-Nearest Neighbours
# ---------------------------------------------------------------------------

"""
TPE search space for MatrixKNNRegressor:
  K            ~ QuantUniform(1, 25, step=1)   number of neighbours
  weights_idx  ~ Choice([0, 1])                0=Uniform, 1=InverseDistance
  metric_idx   ~ Choice([0, 1])                0=Euclidean, 1=Cityblock
"""
function _tpe_space(::KNNSpec)
    Dict{Symbol,Any}(
        :K => HP.QuantUniform(:K, 1.0, 25.0, 1.0),
        :weights_idx => HP.Choice(:weights_idx, [0.0, 1.0]),
        :metric_idx => HP.Choice(:metric_idx, [0.0, 1.0]),
    )
end

function _build_model(::KNNSpec, params::Dict, ::Integer; nthreads::Int=Threads.nthreads())
    MatrixKNNRegressor(
        Int(round(params[:K])),
        Int(round(params[:weights_idx])),
        Int(round(params[:metric_idx])),
    )
end

function _params_to_dict(::KNNSpec, params::Dict)
    Dict{String,Any}(
        "K" => Int(round(params[:K])),
        "weights" => ["uniform", "inverse"][Int(round(params[:weights_idx]))+1],
        "metric" => ["euclidean", "cityblock"][Int(round(params[:metric_idx]))+1],
    )
end

# ---------------------------------------------------------------------------
# Support Vector Regression (RBF kernel)
# ---------------------------------------------------------------------------

"""
TPE search space for EpsilonSVR (RBF kernel):
  cost    ~ LogUniform(log(0.01), log(100))   regularisation C
  gamma   ~ LogUniform(log(1e-4), log(10))    RBF bandwidth
  epsilon ~ LogUniform(log(1e-3), log(1.0))   ε-tube half-width
"""
function _tpe_space(::SVRSpec)
    Dict{Symbol,Any}(
        :cost => HP.LogUniform(:cost, log(0.01), log(100.0)),
        :gamma => HP.LogUniform(:gamma, log(1e-4), log(10.0)),
        :epsilon => HP.LogUniform(:epsilon, log(1e-3), log(1.0)),
    )
end

function _build_model(::SVRSpec, params::Dict, ::Integer; nthreads::Int=Threads.nthreads())
    ThreadedEpsilonSVR(
        cost=Float64(params[:cost]),
        gamma=Float64(params[:gamma]),
        epsilon=Float64(params[:epsilon]),
        nt=nthreads,
    )
end

function _params_to_dict(::SVRSpec, params::Dict)
    Dict{String,Any}(
        "cost" => Float64(params[:cost]),
        "gamma" => Float64(params[:gamma]),
        "epsilon" => Float64(params[:epsilon]),
    )
end

# ---------------------------------------------------------------------------
# Elastic Net
# ---------------------------------------------------------------------------

"""
TPE search space for ElasticNetRegressor:
  lambda ~ LogUniform(log(1e-4), log(10))   overall regularisation strength
  gamma  ~ Uniform(0, 1)                    L1/L2 mixing (0=Ridge, 1=Lasso)
"""
function _tpe_space(::ElasticNetSpec)
    Dict{Symbol,Any}(
        :lambda => HP.LogUniform(:lambda, log(1e-4), log(10.0)),
        :gamma => HP.Uniform(:gamma, 0.0, 1.0),
    )
end

function _build_model(::ElasticNetSpec, params::Dict, ::Integer; nthreads::Int=Threads.nthreads())
    _ElNetRegressor(
        lambda=Float64(params[:lambda]),
        gamma=Float64(params[:gamma]),
    )
end

function _params_to_dict(::ElasticNetSpec, params::Dict)
    Dict{String,Any}(
        "lambda" => Float64(params[:lambda]),
        "gamma" => Float64(params[:gamma]),
    )
end
