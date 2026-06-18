#!/usr/bin/env julia
#
# append_to_arrow.jl — append new Arrow shards to an existing Arrow file.
#
# Reads the existing Arrow file and the new shard files, concatenates them,
# and atomically replaces the existing file (write to .tmp → mv).
#
# Usage:
#   julia --project=. scripts/append_to_arrow.jl <existing_arrow> @<shards_list_file>
#   julia --project=. scripts/append_to_arrow.jl <existing_arrow> <shard1> [shard2 ...]

using Arrow
using Tables

function main(existing_path, new_shard_paths)
    isempty(new_shard_paths) && error("No new shard paths provided")
    isfile(existing_path)    || error("Existing Arrow file not found: $existing_path")

    @info "Appending $(length(new_shard_paths)) shards → $existing_path"

    all_paths = [existing_path; new_shard_paths]
    tmp_path  = existing_path * ".tmp"

    Arrow.write(
        tmp_path,
        Tables.partitioner(Arrow.Table(p) for p in all_paths);
        compress = :zstd,
    )
    mv(tmp_path, existing_path; force=true)

    @info "Done" path=existing_path total_sources=length(all_paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: append_to_arrow.jl <existing_arrow> @<shards_list_file>")
        println("       append_to_arrow.jl <existing_arrow> <shard1> [shard2 ...]")
        exit(1)
    end
    existing_path = ARGS[1]
    rest          = ARGS[2:end]
    new_shard_paths = if length(rest) == 1 && startswith(rest[1], "@")
        readlines(rest[1][2:end])
    else
        rest
    end
    main(existing_path, new_shard_paths)
end
