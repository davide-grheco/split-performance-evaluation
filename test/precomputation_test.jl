using Test
using JLD2
using DataSplitBench
using DataSplits

# ---------------------------------------------------------------------------
# Helpers — mirror the private path helpers in Precomputation.jl
# ---------------------------------------------------------------------------
_methdir(root, dataset, method) = joinpath(root, dataset, method)
_cpath(root, dataset, method)   = joinpath(_methdir(root, dataset, method), "splits.jld2")
_grp(rep, fold)                 = "rep$(rep)_fold$(fold)"

function make_cv_file(root, dataset; k=2, repeats=2, n=20, seed=1)
    splits = DataSplitBench.repeated_cv_splits(n, k, repeats; seed=seed)
    meta   = Dict("n_obs"=>n, "k"=>k, "repeats"=>repeats, "seed"=>seed,
                  "timestamp"=>"test")
    DataSplitBench.save_cv(root, dataset, splits; meta)
    return splits
end

# Write the old-format individual JLD2 files (one per rep/fold)
function write_shards(methdir, reps, folds)
    mkpath(methdir)
    for r in 1:reps, f in 1:folds
        path = joinpath(methdir, "rep$(r)_fold$(f).jld2")
        jldopen(path, "w") do jf
            jf["split"] = (train=[r, f, 1], test=[r+10, f+10], repeat=r, fold=f)
            jf["meta"]  = Dict("rep"=>r, "fold"=>f)
        end
    end
end

# Run the same logic as consolidate_splits.jl inline
const SHARD_RE = r"^rep(\d+)_fold(\d+)\.jld2$"
function consolidate!(methdir)
    shards = filter(f -> occursin(SHARD_RE, f), readdir(methdir))
    isempty(shards) && return 0
    cpath = joinpath(methdir, "splits.jld2")
    n = 0
    jldopen(cpath, "a+") do jf
        for fname in shards
            m = match(SHARD_RE, fname); m === nothing && continue
            rep, fld = parse(Int, m[1]), parse(Int, m[2])
            grp = _grp(rep, fld)
            haskey(jf, grp) && continue
            src = jldopen(joinpath(methdir, fname), "r") do sf
                (split=sf["split"], meta=sf["meta"])
            end
            jf["$grp/split"] = src.split
            jf["$grp/meta"]  = src.meta
            n += 1
        end
    end
    for fname in shards; rm(joinpath(methdir, fname); force=true); end
    return n
end

@testset "Precomputation — consolidated format" begin

    mktempdir() do root
        dataset = "TEST"
        method  = "random"
        reps, folds = 2, 2

        make_cv_file(root, dataset; k=folds, repeats=reps)
        mdir = _methdir(root, dataset, method)

        @testset "migration: shards → consolidated" begin
            write_shards(mdir, reps, folds)
            # Verify shards exist before migration
            @test length(filter(f -> occursin(SHARD_RE, f), readdir(mdir))) == reps * folds

            n = consolidate!(mdir)
            @test n == reps * folds

            # Shards removed, consolidated file present
            @test isfile(_cpath(root, dataset, method))
            @test isempty(filter(f -> occursin(SHARD_RE, f), readdir(mdir)))
        end

        @testset "load_method_split round-trip" begin
            for r in 1:reps, f in 1:folds
                sp = DataSplitBench.load_method_split(root, dataset, method, r, f)
                @test sp isa DataSplits.TrainTestSplit
                # train indices were written as [r, f, 1]
                @test sp.train == [r, f, 1]
                @test sp.test  == [r+10, f+10]
            end
        end

        @testset "each_split iteration" begin
            pairs = collect(DataSplitBench.each_split(root, dataset, method))
            @test length(pairs) == reps * folds
            for p in pairs
                @test haskey(p, :outer)
                @test haskey(p, :inner)
                r, f = p.inner.repeat, p.inner.fold
                @test 1 <= r <= reps
                @test 1 <= f <= folds
                @test p.inner.train == [r, f, 1]
            end
        end

        @testset "migration idempotent" begin
            # Re-running consolidation on already-migrated data is a no-op
            n = consolidate!(mdir)
            @test n == 0
            @test isfile(_cpath(root, dataset, method))
        end

        @testset "load_method_split errors on missing file" begin
            @test_throws AssertionError DataSplitBench.load_method_split(
                root, dataset, "nonexistent_method", 1, 1)
        end

        @testset "load_method_split errors on missing group" begin
            @test_throws AssertionError DataSplitBench.load_method_split(
                root, dataset, method, 99, 99)
        end
    end
end

# ---------------------------------------------------------------------------
# load_split(path, rep, fold) — consolidated experiment splits
# ---------------------------------------------------------------------------
@testset "load_split consolidated (experiment format)" begin
    mktempdir() do dir
        cpath = joinpath(dir, "splits.jld2")

        # Write a consolidated file mimicking consolidate_experiment_splits.jl output
        jldopen(cpath, "w") do jf
            for r in 1:2, f in 1:2
                grp = "rep$(r)_fold$(f)"
                jf["$grp/split"] = (train=[r, f], test=[r+10, f+10], repeat=r, fold=f)
                jf["$grp/meta"]  = Dict("rep"=>r, "fold"=>f)
            end
        end

        @testset "round-trip" begin
            for r in 1:2, f in 1:2
                sp = DataSplitBench.load_split(cpath, r, f)
                @test sp.train == [r, f]
                @test sp.test  == [r+10, f+10]
            end
        end

        @testset "error on missing file" begin
            @test_throws AssertionError DataSplitBench.load_split(
                joinpath(dir, "missing.jld2"), 1, 1)
        end

        @testset "error on missing group" begin
            @test_throws AssertionError DataSplitBench.load_split(cpath, 9, 9)
        end
    end
end
