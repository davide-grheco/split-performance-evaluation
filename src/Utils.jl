using Distances
using MLUtils

export bitvectors_to_bitmatrix, get_data_dir, find_most_distant_pair,
    parse_distance_metric, parse_linkage, parse_umap_metric

bitvectors_to_bitmatrix(vectors::AbstractVector{<:AbstractVector{Bool}})::BitMatrix = hcat(vectors...)

"""
    get_data_dir() -> String

Get the absolute path to the package's data directory, works even when package is dev'ed.
"""
function get_data_dir()
    path = abspath(joinpath(@__DIR__, "..", "Data"))
    isdir(path) || throw(ArgumentError("could not locate package data directory at $path"))
    return path
end

"""
    get_data_path(filename::String) -> String

Get absolute path to a data file with validation.
"""
function get_data_path(filename::String)
    data_dir = get_data_dir()
    filepath = joinpath(data_dir, filename)
    isfile(filepath) || error("Data file $filename not found in $data_dir")
    return filepath
end

"""
    find_most_distant_pair(data; metric)

Return the indices (i, j) of the pair in `data` with the largest distance,
according to the provided `metric`.

Supports both matrices (columns = samples) and vectors of vectors (e.g. fingerprints).
"""
function find_most_distant_pair(data; metric=Distances.Euclidean())
    N = numobs(data)
    sample = i -> getobs(data, i)
    dist = (a, b) -> Distances.evaluate(metric, a, b)

    maxd, i₁, i₂ = -Inf, 0, 0
    for i in 1:N-1, j in i+1:N
        d = dist(sample(i), sample(j))
        if d > maxd
            maxd, i₁, i₂ = d, i, j
        end
    end
    return i₁, i₂
end

const _DISTANCE_METRICS = Dict{String,Distances.SemiMetric}(
    "euclidean" => Distances.Euclidean(),
    "cityblock" => Distances.Cityblock(),
    "manhattan" => Distances.Cityblock(),
    "cosine" => Distances.CosineDist(),
    "chebyshev" => Distances.Chebyshev(),
    "sqeuclidean" => Distances.SqEuclidean(),
    "jaccard" => Distances.Jaccard(),
)

const _LINKAGE_SYMBOLS = Dict{String,Symbol}(
    "ward" => :ward,
    "single" => :single,
    "complete" => :complete,
    "average" => :average,
    "median" => :median,
    "centroid" => :centroid,
)

const _UMAP_METRICS = Dict{String,Symbol}(
    "euclidean" => :euclidean,
    "manhattan" => :manhattan,
    "cityblock" => :manhattan,
    "cosine" => :cosine,
    "hamming" => :hamming,
    "jaccard" => :jaccard,
)

function parse_distance_metric(name::AbstractString)::Distances.SemiMetric
    n = lowercase(strip(name))
    m = get(_DISTANCE_METRICS, n, nothing)
    m !== nothing || throw(ArgumentError("unknown metric \"$name\". Available: $(join(sort(collect(keys(_DISTANCE_METRICS))), ", "))"))
    return m
end

function parse_linkage(name::AbstractString)::Symbol
    n = lowercase(strip(name))
    s = get(_LINKAGE_SYMBOLS, n, nothing)
    s !== nothing || error("Unknown linkage \"$name\". Available: $(join(sort(collect(keys(_LINKAGE_SYMBOLS))), ", "))")
    return s
end

parse_umap_metric(m::Symbol) = m
function parse_umap_metric(name::AbstractString)::Symbol
    n = lowercase(strip(name))
    s = get(_UMAP_METRICS, n, nothing)
    s !== nothing || error("Unknown UMAP metric \"$name\". Available: $(join(sort(collect(keys(_UMAP_METRICS))), ", "))")
    return s
end
