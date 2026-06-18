using Test
using Clustering
using Distances
using OffsetArrays
import DataSplitBench: neighbour_lists

# import your butina function & result type
# using .ButinaClustering: butina, ButinaResult

# 1a) Simple “two perfect clusters” case
@testset "Butina Clustering" begin

    @testset "Butina" begin
        fps1 = [
            BitVector([1, 1, 0, 0, 0, 0]),
            BitVector([1, 1, 0, 0, 0, 0]),
            BitVector([0, 0, 1, 1, 0, 0]),
            BitVector([0, 0, 1, 1, 0, 0]),
        ]
        # with cutoff=1.0 (i.e. Jaccard distance ≤ 0.0 => identical only)
        r1 = butina(fps1; threshold=0.0, metric=Jaccard())
        @test nclusters(r1) == 2
        @test sort(counts(r1)) == [2, 2]

        # with a loose cutoff so identical bits group
        r2 = butina(fps1; threshold=1.0, metric=Jaccard())
        @test nclusters(r2) == 1
        @test sort(counts(r2)) == [4]

        # 1b) Edge‐case: all points identical
        fps2 = fill(BitVector([1, 0, 1, 0, 1, 0]), 5)
        r3 = butina(fps2; threshold=1.0, metric=Jaccard())
        @test nclusters(r3) == 1
        @test counts(r3)[1] == 5

        # 1c) Edge‐case: no two points similar enough
        fps3 = [
            BitVector([1, 0, 0, 0]),
            BitVector([0, 1, 0, 0]),
            BitVector([0, 0, 1, 0]),
        ]
        r4 = butina(fps3; threshold=0.0, metric=Jaccard())  # only identical bits cluster
        @test nclusters(r4) == 3
        @test all(c -> c == 1, counts(r4))
    end

    @testset "neighbour list — computation" begin
        fps = [
            BitVector([1, 0, 0]),
            BitVector([1, 0, 0]),
            BitVector([0, 1, 0])
        ]

        res = neighbour_lists(fps; threshold=0.0, metric=Jaccard())
        nbrs, pos2idx = res.lists, res.map

        @test pos2idx == [1, 2, 3]

        @test nbrs[1] == [2] && nbrs[2] == [1]
        @test isempty(nbrs[3])


        fps_native = [
            BitVector([1, 1, 0]),
            BitVector([0, 1, 1]),
            BitVector([1, 1, 0])
        ]
        fps = OffsetArray(fps_native, -1:1)

        res = neighbour_lists(fps; threshold=0.0, metric=Jaccard())
        nbrs, pos2idx = res.lists, res.map

        @test pos2idx == [-1, 0, 1]

        pos_first = findfirst(==(-1), pos2idx)
        pos_third = findfirst(==(1), pos2idx)

        @test pos_third in nbrs[pos_first]
        @test pos_first in nbrs[pos_third]


        N = 10
        fps = [BitVector(rand(Bool, 32)) for _ in 1:N]

        thr = 0.25
        res = neighbour_lists(fps; threshold=thr, metric=Jaccard())
        nbrs, _ = res.lists, res.map

        for i in 1:N, j in nbrs[i]
            @test i in nbrs[j]
            d = Distances.evaluate(Jaccard(), fps[i], fps[j])
            @test d ≤ thr
        end
    end

    @testset "merge_small_clusters" begin
        # Test data - two clear clusters and some noise
        fps = [
            BitVector([1, 1, 0, 0, 0, 0]),  # Cluster 1
            BitVector([1, 1, 0, 0, 0, 0]),
            BitVector([0, 0, 1, 1, 0, 0]),  # Cluster 2
            BitVector([0, 0, 1, 1, 0, 0]),
            BitVector([1, 0, 0, 0, 0, 0]),  # Noise points
            BitVector([0, 0, 1, 0, 0, 0]),
            BitVector([0, 0, 0, 0, 1, 1])   # Isolated point
        ]

        # First run Butina clustering with tight threshold
        butina_result = butina(fps; threshold=0.0, metric=Jaccard())

        # Merge clusters smaller than 2 points
        merged = merge_small_clusters(butina_result, fps, 2; metric=Jaccard())

        # Should have merged the single-point clusters into nearest neighbors
        @test nclusters(merged) == 2
        @test sort(counts(merged)) == [3, 4]  # One cluster absorbed the noise points

        merged = merge_small_clusters(butina_result, fps, 1; metric=Jaccard())
        @test merged == butina_result  # No changes expected

        merged = merge_small_clusters(butina_result, fps, 2; metric=Jaccard())
        @test nclusters(merged) == 2
        @test counts(merged) == [4, 3]

        # Should merge small clusters to nearest by Jaccard distance
        small_fps = [
            BitVector([1, 1, 0, 0]),  # Cluster A
            BitVector([1, 1, 0, 0]),
            BitVector([1, 0, 0, 0]),  # Should merge with A (distance 0.33)
            BitVector([0, 0, 1, 1]),  # Cluster B
            BitVector([0, 0, 1, 0])   # Should merge with B (distance 0.5)
        ]

        butina_res = butina(small_fps; threshold=0.0, metric=Jaccard())
        merged = merge_small_clusters(butina_res, small_fps, 2; metric=Jaccard())

        # Verify points merged to correct clusters
        assignments = merged.assignments
        @test assignments[3] == assignments[1]
        @test assignments[5] == assignments[4]

        butina_res = butina(fps[5:7]; threshold=0.0, metric=Jaccard())
        @test all(c == 1 for c in counts(butina_res))

        merged = merge_small_clusters(butina_res, fps[5:7], 2; metric=Jaccard())
        @test merged == butina_res
    end

    @testset "butina — BitMatrix input" begin
        using DataSplitBench: bitvectors_to_bitmatrix
        fps = [
            BitVector([1, 1, 0, 0]),
            BitVector([1, 1, 0, 0]),
            BitVector([0, 0, 1, 1]),
            BitVector([0, 0, 1, 1]),
        ]
        X = bitvectors_to_bitmatrix(fps)   # 4 features × 4 samples
        result_vec    = butina(fps;  threshold=0.2, metric=Jaccard())
        result_matrix = butina(X;    threshold=0.2, metric=Jaccard())
        @test result_vec.assignments == result_matrix.assignments
        @test result_vec.counts      == result_matrix.counts
    end
end
