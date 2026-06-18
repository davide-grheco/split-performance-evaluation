#!/usr/bin/env julia
#
# compress_existing.jl — retroactively compress Arrow and JLD2 files in-place.
#
# Writes a compressed copy and replaces the original only when it is at least
# SAVE_THRESHOLD smaller, so already-compressed files are left untouched.
# Both formats decompress transparently on read, so the result is fully
# backward-compatible with uncompressed files.
#
# Usage:
#   julia --project=. scripts/compress_existing.jl [--dry-run] [dir|file ...]
#   Defaults to "." when no paths are given.

using Arrow, JLD2

const SAVE_THRESHOLD = 0.95   # replace only when new_size / orig_size < this

# ---------------------------------------------------------------------------
# Arrow
# ---------------------------------------------------------------------------

function compress_arrow!(path::String; dry_run::Bool)::Bool
    islink(path) && return false
    orig_size = filesize(path)
    tmp = path * ".compress.$(getpid()).tmp"
    try
        tbl = Arrow.Table(path)
        Arrow.write(tmp, tbl; compress=:zstd)
        new_size = filesize(tmp)
        if new_size < orig_size * SAVE_THRESHOLD
            dry_run || mv(tmp, path; force=true)
            pct = round(Int, 100 * (1 - new_size / orig_size))
            @info (dry_run ? "[dry] Arrow" : "Arrow") path savings="$(pct)%"
            return true
        end
        return false
    catch e
        @error "Arrow compression failed" path exception=e
        return false
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
end

# ---------------------------------------------------------------------------
# JLD2
# ---------------------------------------------------------------------------

# Returns a flat Dict of all user-readable entries in a JLD2 file.
# Handles both flat keys ("split", "meta") and one-level groups
# ("rep1_fold1/split", "rep1_fold1/meta") which cover all structures in
# this project.  Groups are discovered via the underlying HDF5 handle to
# avoid JLD2 trying to deserialize a plain group as a Julia type.
function collect_jld2_entries(src::JLD2.JLDFile)::Dict{String,Any}
    entries = Dict{String,Any}()
    for k1 in keys(src)
        k1 == "_types" && continue   # JLD2-internal type registry
        try
            entries[k1] = src[k1]
        catch
            # k1 is a plain HDF5 group with no julia_type attribute —
            # descend one level via the HDF5 handle.
            hgrp = try src.plain[k1]; catch; continue; end
            try
                for k2 in keys(hgrp)
                    startswith(k2, "_") && continue
                    try
                        entries["$k1/$k2"] = src["$k1/$k2"]
                    catch e2
                        @debug "Skipping nested key" path="$k1/$k2" exception=e2
                    end
                end
            finally
                close(hgrp)
            end
        end
    end
    return entries
end

function compress_jld2!(path::String; dry_run::Bool)::Bool
    islink(path) && return false
    orig_size = filesize(path)
    tmp = path * ".compress.$(getpid()).tmp"
    try
        entries = jldopen(path, "r") do src
            collect_jld2_entries(src)
        end
        isempty(entries) && return false

        jldopen(tmp, "w"; compress=true) do dst
            for (k, v) in entries
                dst[k] = v
            end
        end

        new_size = filesize(tmp)
        if new_size < orig_size * SAVE_THRESHOLD
            dry_run || mv(tmp, path; force=true)
            pct = round(Int, 100 * (1 - new_size / orig_size))
            @info (dry_run ? "[dry] JLD2" : "JLD2") path savings="$(pct)%"
            return true
        end
        return false
    catch e
        @error "JLD2 compression failed" path exception=e
        return false
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(args::Vector{String})
    dry_run = "--dry-run" in args
    roots   = filter(!=("--dry-run"), args)
    isempty(roots) && push!(roots, ".")

    n_arrow_done = n_arrow_skip = 0
    n_jld2_done  = n_jld2_skip  = 0

    function process(path::String)
        if endswith(path, ".arrow")
            if compress_arrow!(path; dry_run)
                n_arrow_done += 1
            else
                n_arrow_skip += 1
            end
        elseif endswith(path, ".jld2")
            if compress_jld2!(path; dry_run)
                n_jld2_done += 1
            else
                n_jld2_skip += 1
            end
        end
    end

    for root in roots
        if isfile(root)
            process(root)
        elseif isdir(root)
            for (dir, _, files) in walkdir(root)
                for fname in files
                    process(joinpath(dir, fname))
                end
            end
        else
            @warn "Path not found, skipping" path=root
        end
    end

    @info "Done" arrow_compressed=n_arrow_done arrow_skipped=n_arrow_skip \
                  jld2_compressed=n_jld2_done  jld2_skipped=n_jld2_skip
end

main(ARGS)
