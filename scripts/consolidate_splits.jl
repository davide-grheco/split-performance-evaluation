#!/usr/bin/env julia
# Migrate Splits/ from per-rep-fold JLD2 files to one consolidated splits.jld2
# per (dataset, method). Safe to re-run: already-migrated groups are skipped.
#
# Usage:
#   julia --project=. scripts/consolidate_splits.jl [splits_root]
#
# Default splits_root: Splits/merck

using JLD2

const SHARD_RE = r"^rep(\d+)_fold(\d+)\.jld2$"

function consolidate_method_dir(methdir::String)
    shards = filter(f -> occursin(SHARD_RE, f), readdir(methdir))
    isempty(shards) && return 0

    cpath = joinpath(methdir, "splits.jld2")
    migrated = 0

    jldopen(cpath, "a+") do jf
        for fname in shards
            m = match(SHARD_RE, fname)
            m === nothing && continue
            rep  = parse(Int, m[1])
            fld  = parse(Int, m[2])
            grp  = "rep$(rep)_fold$(fld)"
            haskey(jf, grp) && continue  # already migrated

            fpath = joinpath(methdir, fname)
            split_data, meta_data = jldopen(fpath, "r") do sf
                sf["split"], sf["meta"]
            end
            jf["$grp/split"] = split_data
            jf["$grp/meta"]  = meta_data
            migrated += 1
        end
    end

    # Delete originals only after all groups are safely written
    for fname in shards
        rm(joinpath(methdir, fname); force=true)
    end

    return migrated
end

function main(splits_root::String)
    @assert isdir(splits_root) "Not a directory: $splits_root"

    total_methods = 0
    total_shards  = 0

    for dataset_entry in readdir(splits_root; join=true)
        isdir(dataset_entry) || continue
        dataset = basename(dataset_entry)
        dataset == "cv" && continue

        for method_entry in readdir(dataset_entry; join=true)
            isdir(method_entry) || continue
            method = basename(method_entry)
            method == "cv" && continue

            n = consolidate_method_dir(method_entry)
            if n > 0
                @info "Consolidated" dataset method shards=n
                total_shards  += n
                total_methods += 1
            end
        end
    end

    @info "Done" methods_consolidated=total_methods total_shards_merged=total_shards
end

root = length(ARGS) >= 1 ? ARGS[1] : "Splits/merck"
main(root)
