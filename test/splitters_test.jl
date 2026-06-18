using Test
using Dates
using Random
using DataSplitBench
using MLUtils

@testset "Splitters" begin

    @testset "time_split" begin
        dates = Date.(["2020-01-01", "2020-06-01", "2021-01-01",
                       "2021-06-01", "2022-01-01"])
        split = time_split(dates; frac=0.6)
        train, test = split.train, split.test

        @test length(train) + length(test) == length(dates)
        @test sort(vcat(train, test)) == collect(1:length(dates))
        @test isempty(intersect(train, test))
        # cutoff is sorted_dates[split_at]; train gets everything strictly before it,
        # so train has exactly split_at - 1 elements (with unique dates)
        split_at = floor(Int, 0.6 * length(dates))
        @test length(train) == split_at - 1
        @test length(test)  == length(dates) - split_at + 1
        # all train dates strictly before all test dates
        @test all(dates[i] < dates[j] for i in train, j in test)
    end

    @testset "random_split" begin
        rng = MersenneTwister(42)
        X = randn(rng, 5, 20)   # 5 features × 20 samples
        split = random_split(X; frac=0.8, rng=MersenneTwister(1))
        train, test = split.train, split.test

        @test length(train) + length(test) == numobs(X)
        @test sort(vcat(train, test)) == collect(1:numobs(X))
        @test isempty(intersect(train, test))
        @test length(train) == floor(Int, 0.8 * numobs(X))
        # all indices are valid
        @test all(1 .<= train .<= numobs(X))
        @test all(1 .<= test  .<= numobs(X))
    end

    @testset "butina_split — BitMatrix input" begin
        fps = [
            BitVector([1, 1, 0, 0]),
            BitVector([1, 1, 0, 0]),
            BitVector([0, 0, 1, 1]),
            BitVector([0, 0, 1, 1]),
            BitVector([1, 1, 1, 1]),
        ]
        X = bitvectors_to_bitmatrix(fps)   # 4 features × 5 samples
        @test size(X) == (4, 5)

        split = butina_split(X; cutoff=0.5, frac=0.6, rng=MersenneTwister(123))
        train, test = split.train, split.test

        @test length(train) + length(test) == numobs(X)
        @test sort(vcat(train, test)) == collect(1:numobs(X))
        @test isempty(intersect(train, test))
        @test length(train) >= floor(Int, 0.6 * numobs(X))
    end
end
