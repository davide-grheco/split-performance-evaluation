using JLD2, JSON3

export load_method_split, load_split

"""
    load_split(path) -> NamedTuple (train, test, ...)

Load the JLD2 split file written by `run_split.jl`. Returns the `"split"` entry.
"""
function load_split(path::String)
    jldopen(path, "r") do f
        f["split"]
    end
end

"""
    load_split(path, rep, fold) -> NamedTuple (train, test, ...)

Load one (rep, fold) entry from a consolidated `splits.jld2` produced by
`consolidate_experiment_splits.jl`.
"""
function load_split(path::String, rep::Int, fold::Int)
    grp = _split_group(rep, fold)
    @assert isfile(path) "Missing consolidated split file at $path"
    jldopen(path, "r") do f
        @assert haskey(f, grp) "No group $grp in $path"
        f["$grp/split"]
    end
end

# ---------- 3) Paths & I/O ----------
_base(root, dataset) = joinpath(root, dataset)
_cvdir(root, dataset) = joinpath(_base(root, dataset), "cv")
_cvpath(root, dataset) = joinpath(_cvdir(root, dataset), "cv_splits.jld2")
_cvmeta(root, dataset) = joinpath(_cvdir(root, dataset), "meta.json")
_methdir(root, dataset, method_id) = joinpath(root, dataset, method_id)

"Path to the consolidated JLD2 for a (dataset, method) pair."
_consolidated_path(root, dataset, method_id) =
    joinpath(_methdir(root, dataset, method_id), "splits.jld2")

"Group key for a (rep, fold) inside the consolidated JLD2."
_split_group(rep::Int, fold::Int) = "rep$(rep)_fold$(fold)"

"Save outer CV once."
function save_cv(root::AbstractString, dataset::AbstractString, splits; meta::Dict)
    mkpath(_cvdir(root, dataset))
    jldopen(_cvpath(root, dataset), "w"; compress=true) do f
        f["splits"] = splits
        f["meta"] = meta
    end
    # duplicate light meta as JSON for quick inspection
    open(_cvmeta(root, dataset), "w") do io
        JSON3.write(io, meta)
    end
    return nothing
end

"Load outer CV."
function load_cv(root::AbstractString, dataset::AbstractString)
    path = _cvpath(root, dataset)
    @assert isfile(path) "Missing CV file at $path"
    splits = nothing
    meta = Dict{String,Any}()
    jldopen(path, "r") do f
        splits = f["splits"]
        meta = Dict{String,Any}(f["meta"])
    end
    return splits, meta
end

function load_cv(root::AbstractString, dataset::AbstractString, repeat::Int, fold::Int)::DataSplits.TrainTestSplit
    splits, _ = load_cv(root, dataset)

    idx = findfirst(s -> s.repeat == repeat && s.fold == fold, splits)
    idx === nothing && throw(ArgumentError("no CV split for repeat=$repeat fold=$fold"))
    split = splits[idx]
    train, test = split.train, split.test

    return DataSplits.TrainTestSplit(train, test)
end

# ---------- 4) Builders ----------
"""
    build_cv_for_dataset(; store_root, dataset, n_obs, k, repeats, seed)

Create/overwrite outer CV for a dataset.
"""
function build_cv_for_dataset(; store_root::AbstractString, dataset::AbstractString,
    n_obs::Int, k::Int, repeats::Int, seed::Union{Nothing,Int}=nothing)
    splits = repeated_cv_splits(n_obs, k, repeats; seed=seed)
    meta = Dict(
        "n_obs" => n_obs, "k" => k, "repeats" => repeats,
        "seed" => something(seed, "nothing"), "timestamp" => string(Dates.now())
    )
    save_cv(store_root, dataset, splits; meta)
    return splits, meta
end

# ---------- 5) Loading / iterating ----------
"Load one inner split (train/val) for a given dataset/method/repeat/fold."
function load_method_split(root::AbstractString, dataset::AbstractString, method_id::AbstractString, rep::Int, fold::Int)
    cpath = _consolidated_path(root, dataset, method_id)
    grp = _split_group(rep, fold)
    @assert isfile(cpath) "Missing consolidated split file at $cpath"
    s = jldopen(cpath, "r") do f
        @assert haskey(f, grp) "No group $grp in $cpath"
        f["$grp/split"]
    end
    return DataSplits.TrainTestSplit(s.train, s.test)
end

"Iterate aligned outer+inner splits."
function each_split(root::AbstractString, dataset::AbstractString, method_id::AbstractString)
    cv, _ = load_cv(root, dataset)
    cpath = _consolidated_path(root, dataset, method_id)
    @assert isfile(cpath) "Missing consolidated split file at $cpath"
    return Iterators.map(cv) do s
        grp = _split_group(s[:repeat], s[:fold])
        inner = jldopen(cpath, "r") do f
            @assert haskey(f, grp) "No group $grp in $cpath"
            f["$grp/split"]
        end
        return (outer=s, inner=inner)
    end
end
