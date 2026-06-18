#!/usr/bin/env julia
#
# subsample_cv.jl — pre-compute a reproducible random subsample of one outer-CV fold.
#
# For each (dataset, rep, fold), draws `subsample_n` global indices at random from
# the outer-train partition and saves them to a JLD2 file.  All splitters that run
# on the same (rep, fold) pair load these same indices, ensuring a fair comparison
# across splitting methods.
#
# Usage (called by Snakemake):
#   julia --project=. scripts/subsample_cv.jl \
#       <dataset> <cv_jld2> <rep> <fold> <subsample_n> <seed> <out_jld2>

using DataSplitBench
using Random
using JLD2
using Dates

function main(dataset, _cv_jld2, rep, fold, subsample_n, seed, out_path)
    t0 = Dates.now()
    @info "Starting" dataset rep fold timestamp=t0
    roots      = DataSplitBench.Config.load_experiment_roots()
    split_root = roots.split_root

    cv              = DataSplitBench.load_cv(split_root, dataset, rep, fold)
    outer_train_idx = cv.train   # global indices into the full training set

    n_available = length(outer_train_idx)
    n_draw      = min(subsample_n, n_available)
    n_draw < subsample_n && @warn "Only $n_available outer-train samples for $dataset (rep=$rep fold=$fold); drawing all"

    rng        = MersenneTwister(seed)
    sampled    = sort(shuffle(rng, outer_train_idx)[1:n_draw])

    mkpath(dirname(out_path))
    jldopen(out_path, "w"; compress=true) do f
        f["idx"]  = sampled
        f["meta"] = Dict(
            "dataset"     => dataset,
            "rep"         => rep,
            "fold"        => fold,
            "subsample_n" => subsample_n,
            "n_available" => n_available,
            "n_drawn"     => n_draw,
            "seed"        => seed,
            "timestamp"   => string(Dates.now()),
        )
    end

    elapsed = round(Dates.now() - t0, Dates.Second)
    @info "Done" dataset rep fold n_available n_drawn=n_draw path=out_path elapsed=elapsed
    println("Wrote subsample rep=$rep fold=$fold (n=$n_draw) → $out_path  [$(elapsed)]")
end

if length(ARGS) < 7
    println("Usage: subsample_cv.jl <dataset> <cv_jld2> <rep> <fold> <subsample_n> <seed> <out_jld2>")
    exit(1)
end

dataset     = ARGS[1]
cv_jld2     = ARGS[2]   # passed for Snakemake dependency tracking
rep         = parse(Int, ARGS[3])
fold        = parse(Int, ARGS[4])
subsample_n = parse(Int, ARGS[5])
seed        = parse(Int, ARGS[6])
out_path    = ARGS[7]

main(dataset, cv_jld2, rep, fold, subsample_n, seed, out_path)
