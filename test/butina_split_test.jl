using Test
using Random
using Distances
using MLUtils

# Import your clustering functions and types; adjust the module name as needed:
# using ButinaClustering: butina, ButinaResult, butina_split


# —————————————————————————————————————————————————————————————————————
@testset "Butina split tests" begin
    # Toy fingerprints including a singleton
    fps = [
        BitVector([1, 1, 0, 0]),
        BitVector([1, 1, 0, 0]),
        BitVector([0, 0, 1, 1]),
        BitVector([0, 0, 1, 1]),
        BitVector([1, 1, 1, 1]),
    ]

    # 80:20 split by whole clusters
    split = butina_split(fps; cutoff=0.5, frac=0.6, rng=MersenneTwister(123))
    train, test = split.train, split.test

    # 1) Coverage & disjointness
    @test length(train) + length(test) == length(fps)
    @test sort(vcat(train, test)) == collect(1:length(fps))
    @test isempty(intersect(train, test))

    # 2) Train size ≥ floor(frac * n)
    @test length(train) ≥ floor(Int, 0.6 * length(fps))
end

@testset "UMAP split tests" begin
    # Toy fingerprints including a singleton
    fps = [
        BitVector([1, 1, 0, 0]),
        BitVector([1, 1, 0, 0]),
        BitVector([0, 0, 1, 1]),
        BitVector([0, 0, 1, 1]),
        BitVector([1, 1, 1, 1]),
    ] |> bitvectors_to_bitmatrix

    # 80:20 split by whole clusters
    split = umap_split(fps; frac=0.6, rng=MersenneTwister(123))
    train, test = split.train, split.test

    # 1) Coverage & disjointness
    @test length(train) + length(test) == numobs(fps)
    @test sort(vcat(train, test)) == collect(1:numobs(fps))
    @test isempty(intersect(train, test))

    # 2) Train size ≥ floor(frac * n)
    @test length(train) ≥ floor(Int, 0.6 * numobs(fps))
end
