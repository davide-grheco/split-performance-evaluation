using Test
using Distances
using DataSplitBench

@testset "Utils" begin

    @testset "bitvectors_to_bitmatrix" begin
        fps = [BitVector([1, 0, 1]), BitVector([0, 1, 0]), BitVector([1, 1, 1])]
        M = bitvectors_to_bitmatrix(fps)
        # features × samples: 3 bits × 3 fingerprints
        @test size(M) == (3, 3)
        @test M[:, 1] == fps[1]
        @test M[:, 2] == fps[2]
        @test M[:, 3] == fps[3]
    end

    @testset "find_most_distant_pair" begin
        @testset "matrix input (columns = samples)" begin
            # columns: [0,0], [1,0], [10,0] — most distant are cols 1 and 3
            X = [0.0 1.0 10.0; 0.0 0.0 0.0]
            i, j = find_most_distant_pair(X)
            @test Set([i, j]) == Set([1, 3])
        end

        @testset "vector-of-vectors input" begin
            data = [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]
            i, j = find_most_distant_pair(data)
            @test Set([i, j]) == Set([1, 3])
        end

        @testset "custom metric" begin
            X = [0.0 1.0 5.0; 0.0 0.0 0.0]
            i, j = find_most_distant_pair(X; metric=Cityblock())
            @test Set([i, j]) == Set([1, 3])
        end
    end

    @testset "parse_distance_metric" begin
        @test parse_distance_metric("euclidean") isa Distances.Euclidean
        @test parse_distance_metric("jaccard") isa Distances.Jaccard
        @test parse_distance_metric("cosine") isa Distances.CosineDist
        @test parse_distance_metric("EUCLIDEAN") isa Distances.Euclidean  # case-insensitive
        @test_throws ArgumentError parse_distance_metric("unknown_metric")
    end

    @testset "parse_linkage" begin
        @test parse_linkage("ward") == :ward
        @test parse_linkage("single") == :single
        @test parse_linkage("complete") == :complete
        @test parse_linkage("WARD") == :ward  # case-insensitive
        @test_throws ErrorException parse_linkage("no_such_linkage")
    end

    @testset "parse_umap_metric" begin
        @test parse_umap_metric("euclidean") == :euclidean
        @test parse_umap_metric("jaccard") == :jaccard
        @test parse_umap_metric(:cosine) == :cosine  # symbol passthrough
        @test_throws ErrorException parse_umap_metric("no_such_metric")
    end
end
