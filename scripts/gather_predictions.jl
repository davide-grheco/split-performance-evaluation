#!/usr/bin/env julia
#
# gather_predictions.jl — concatenate per-(rep, fold) Arrow shards into one
# Arrow file per (dataset, model).
#
# Usage (called by Snakemake):
#   # Pass shard paths via a file (avoids shell argument-length limits):
#   julia --project=. scripts/gather_predictions.jl <out_arrow> @<shards_list_file>
#
#   # Legacy: positional args still work for small shard counts:
#   julia --project=. scripts/gather_predictions.jl <out_arrow> <shard1> [shard2 ...]

using Arrow
using Tables

function main(out_path, shard_paths)
    isempty(shard_paths) && error("No shard paths provided")

    @info "Gathering $(length(shard_paths)) shards → $out_path"

    # Generator keeps at most one shard open at a time; Tables.partitioner
    # drives it one batch at a time so peak memory is O(one shard), not O(all).
    mkpath(dirname(out_path))
    Arrow.write(out_path, Tables.partitioner(Arrow.Table(p) for p in shard_paths); compress=:zstd)

    @info "Done" path=out_path
end

# ---------------------------------------------------------------------------
# CLI — supports both @filelist and positional-arg forms
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: gather_predictions.jl <out_arrow> @<shards_list_file>")
        println("       gather_predictions.jl <out_arrow> <shard1> [shard2 ...]")
        exit(1)
    end
    out_path = ARGS[1]
    rest     = ARGS[2:end]
    # If a single argument starts with '@', treat it as a file listing shard paths.
    shard_paths = if length(rest) == 1 && startswith(rest[1], "@")
        readlines(rest[1][2:end])
    else
        rest
    end
    main(out_path, shard_paths)
end
