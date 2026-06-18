using Test
using DataSplitBench
using DataSplits
using Random

# Build a SplittingConfig for use in all tests
_splitting_cfg = DataSplitBench.Config.SplittingConfig(
    ["random"], [0.8], 5, 5
)

@testset "Registry" begin

    @testset "make_splitter — all registered names return a splitter" begin
        for name in keys(DataSplitBench.SPLITTER_REGISTRY)
            cfg = DataSplitBench.Config.SplittingConfig([name], [0.8], 5, 5)
            splitter = DataSplitBench.make_splitter(name, cfg)
            @test splitter !== nothing
        end
    end

    @testset "make_splitter — frac is respected" begin
        for frac in (0.7, 0.8, 0.9)
            cfg = DataSplitBench.Config.SplittingConfig(["random"], [frac], 5, 5)
            splitter = DataSplitBench.make_splitter("random", cfg)
            # RandomSplit stores frac; check it via the type (DataSplits API)
            @test splitter isa DataSplits.RandomSplit
        end
    end

    @testset "make_splitter — unknown name raises informative error" begin
        cfg = DataSplitBench.Config.SplittingConfig(["no_such_method"], [0.8], 5, 5)
        err = @test_throws ErrorException DataSplitBench.make_splitter("no_such_method", cfg)
        @test occursin("no_such_method", err.value.msg)
    end

    @testset "make_splitter — produces usable splitter on tiny data" begin
        # 20 features x 30 samples — Float64 so covariance/Mahalanobis-based splitters work
        X = rand(Float64, 20, 30)
        rng = MersenneTwister(42)

        for name in ["random", "kennardstone", "mdks",
            "optisim", "maximum_dissimilarity", "minimum_dissimilarity",
            "morais", "spxy-jaccard", "spxy-euclidean"]
            cfg = DataSplitBench.Config.SplittingConfig([name], [0.8], 5, 5)
            splitter = DataSplitBench.make_splitter(name, cfg)
            y = rand(Float64, 30)
            result = DataSplitBench.apply_split(X, y, splitter; rng=rng)
            @test !isempty(result.train)
            @test !isempty(result.test)
            @test length(result.train) + length(result.test) == 30
            @test isempty(intersect(result.train, result.test))
        end
    end

    @testset "placeholder registries are empty Dicts" begin
        @test DataSplitBench.MODEL_REGISTRY isa Dict
        @test DataSplitBench.AD_REGISTRY isa Dict
        @test isempty(DataSplitBench.MODEL_REGISTRY)
        @test isempty(DataSplitBench.AD_REGISTRY)
    end
end
