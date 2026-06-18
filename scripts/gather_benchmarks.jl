#!/usr/bin/env julia
# Consolidate Snakemake per-job benchmark TSV files into one Arrow file.
#
# Reads every .tsv in <subdir>, verifies the count matches --expected,
# concatenates with a `key` column, writes compressed Arrow, deletes TSVs.
#
# Usage:
#   julia --project=. scripts/gather_benchmarks.jl --expected N <subdir> <out.arrow>

using CSV, DataFrames, Arrow

function main(subdir::String, out_path::String, expected::Int)
    isdir(subdir) || error("Benchmark subdir not found: $subdir")

    tsvs = sort(filter(f -> endswith(f, ".tsv"), readdir(subdir; join=true)))

    length(tsvs) == expected ||
        error("Expected $expected benchmark TSVs in $subdir, found $(length(tsvs)). " *
              "Some jobs may have failed or their benchmark files are missing.")

    frames = map(tsvs) do path
        df = CSV.read(path, DataFrame; delim='\t', missingstring="")
        df[!, :key] .= splitext(basename(path))[1]
        df
    end

    combined = vcat(frames...; cols=:union)
    mkpath(dirname(out_path))
    Arrow.write(out_path, combined; compress=:zstd)

    foreach(rm, tsvs)
    @info "Done" out=out_path tsv_count=length(tsvs) rows=nrow(combined)
end

# Parse --expected N flag
args = copy(ARGS)
expected_idx = findfirst(==("--expected"), args)
if expected_idx === nothing || expected_idx == length(args)
    println("Usage: gather_benchmarks.jl --expected N <subdir> <out.arrow>")
    exit(1)
end
expected = parse(Int, args[expected_idx + 1])
deleteat!(args, expected_idx:expected_idx+1)
length(args) == 2 || (println("Usage: gather_benchmarks.jl --expected N <subdir> <out.arrow>"); exit(1))

main(args[1], args[2], expected)
