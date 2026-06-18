#!/usr/bin/env julia

using JLD2
using CodecZlib

const DEFAULT_EXPECTED = 25

function usage()
    println("""
Usage:

  Single output:
    julia --project=. scripts/consolidate_experiment_splits.jl --expected 25 <out.jld2> <shard1.jld2> ...
    julia --project=. scripts/consolidate_experiment_splits.jl --expected 25 <out.jld2> @shardlist.txt

  Scan mode, no shell loop needed:
    julia --project=. scripts/consolidate_experiment_splits.jl --scan experiments/revision_v1/splits --expected 25

Notes:
  - Consolidation only happens when exactly EXPECTED shards are present.
  - Existing splits.jld2 files are skipped if complete.
  - Incomplete existing splits.jld2 files cause an error.
  - Shards are deleted only after the final output has been written and verified.
""")
end

function group_count(path::AbstractString)
    jldopen(path, "r") do jf
        return length(keys(jf))
    end
end

function read_shards(shard_paths::AbstractVector{<:AbstractString}; expected::Int)
    length(shard_paths) == expected ||
        error("Refusing to consolidate: got $(length(shard_paths)) shards, expected $expected")

    groups = Dict{String,Tuple{Any,Any}}()

    for fpath in shard_paths
        isfile(fpath) || error("Shard not found: $fpath")

        split_data, meta_data = jldopen(fpath, "r") do sf
            haskey(sf, "split") || error("Missing key 'split' in shard: $fpath")
            haskey(sf, "meta") || error("Missing key 'meta' in shard: $fpath")
            sf["split"], sf["meta"]
        end

        rep = getproperty(split_data, :repeat)
        fld = getproperty(split_data, :fold)
        grp = "rep$(rep)_fold$(fld)"

        haskey(groups, grp) && error("Duplicate shard group $grp from file: $fpath")
        groups[grp] = (split_data, meta_data)
    end

    length(groups) == expected ||
        error("Refusing to consolidate: got $(length(groups)) unique groups, expected $expected")

    return groups
end

function write_atomic(out_path::AbstractString, groups::Dict{String,Tuple{Any,Any}})
    out_dir = dirname(out_path)
    mkpath(out_dir)

    tmp_path = joinpath(out_dir, ".$(basename(out_path)).tmp.$(getpid())")

    isfile(tmp_path) && rm(tmp_path; force=true)

    try
        jldopen(tmp_path, "w"; compress=true) do jf
            for grp in sort(collect(keys(groups)))
                split_data, meta_data = groups[grp]
                jf["$grp/split"] = split_data
                jf["$grp/meta"] = meta_data
            end
        end

        observed = group_count(tmp_path)
        expected = length(groups)

        observed == expected ||
            error("Internal verification failed for $tmp_path: $observed/$expected groups")

        mv(tmp_path, out_path; force=false)
    catch
        isfile(tmp_path) && rm(tmp_path; force=true)
        rethrow()
    end
end

function consolidate(out_path::AbstractString, shard_paths::AbstractVector{<:AbstractString}; expected::Int)
    if isfile(out_path)
        n = group_count(out_path)
        if n == expected
            @info "Already complete, skipping" out = out_path groups = n
            return false
        else
            error("Existing output is incomplete: $out_path has $n/$expected groups. Delete or restore it first.")
        end
    end

    groups = read_shards(shard_paths; expected=expected)
    write_atomic(out_path, groups)

    n = group_count(out_path)
    n == expected || error("Post-write verification failed: $out_path has $n/$expected groups")

    @info "Done" out = out_path shards_merged = length(shard_paths)
    return true
end

function shard_paths_from_args(args::AbstractVector{<:AbstractString})
    isempty(args) && error("No shard paths provided")

    if length(args) == 1 && startswith(args[1], "@")
        listfile = args[1][2:end]
        isfile(listfile) || error("Shard list not found: $listfile")
        return String.(filter(!isempty, strip.(readlines(listfile))))
    end

    return collect(args)
end

function dirs_with_shards(root::String)
    isdir(root) || error("Root directory not found: $root")

    dirs = String[]

    for (dir, _, files) in walkdir(root)
        any(f -> startswith(f, "rep") && endswith(f, ".jld2"), files) || continue
        push!(dirs, dir)
    end

    return sort(unique(dirs))
end

function scan_and_consolidate(root::String; expected::Int)
    n_done = 0
    n_skip = 0
    n_incomplete = 0

    for dir in dirs_with_shards(root)
        out = joinpath(dir, "splits.jld2")

        if isfile(out)
            n = group_count(out)
            if n == expected
                n_skip += 1
                @info "Already complete, skipping" dir = dir groups = n
                continue
            else
                error("Existing incomplete output found: $out has $n/$expected groups. Restore/delete it before scanning.")
            end
        end

        shards = sort([
            joinpath(dir, f)
            for f in readdir(dir)
            if startswith(f, "rep") && endswith(f, ".jld2")
        ])

        if length(shards) != expected
            n_incomplete += 1
            @info "Incomplete shard set, leaving untouched" dir = dir shards = length(shards) expected = expected
            continue
        end

        @info "Consolidating" dir = dir shards = length(shards)
        consolidate(out, shards; expected=expected) && (n_done += 1)
    end

    @info "Scan done" root = root consolidated = n_done already_complete = n_skip incomplete_left_untouched = n_incomplete
end

function parse_cli(args::Vector{String})
    expected = DEFAULT_EXPECTED
    scan_root = nothing
    rest = String[]

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--expected"
            i += 1
            i <= length(args) || error("--expected requires a value")
            expected = parse(Int, args[i])
        elseif startswith(arg, "--expected=")
            expected = parse(Int, split(arg, "=", limit=2)[2])
        elseif arg == "--scan"
            i += 1
            i <= length(args) || error("--scan requires a root directory")
            scan_root = args[i]
        elseif startswith(arg, "--scan=")
            scan_root = split(arg, "=", limit=2)[2]
        elseif arg in ("-h", "--help")
            usage()
            exit(0)
        else
            push!(rest, arg)
        end

        i += 1
    end

    expected > 0 || error("--expected must be positive")

    return expected, scan_root, rest
end

function main()
    expected, scan_root, rest = parse_cli(ARGS)

    if scan_root !== nothing
        isempty(rest) || error("Unexpected extra arguments in --scan mode: $(join(rest, " "))")
        scan_and_consolidate(scan_root; expected=expected)
        return
    end

    length(rest) >= 2 || begin
        usage()
        exit(1)
    end

    out_path = rest[1]
    shards = shard_paths_from_args(rest[2:end])

    consolidate(out_path, shards; expected=expected)
end

main()
