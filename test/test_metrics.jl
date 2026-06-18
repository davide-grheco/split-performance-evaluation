using Test
using Distances
using MLUtils

@testset "Metrics module" begin

    # tanimoto
    @testset "tanimoto" begin
        a = BitVector([true, false, true])
        b = BitVector([true, true, false])
        @test tanimoto(a, b) ≈ 1 / 3
        @test tanimoto(BitVector([true, false]), BitVector([false, true])) == 0.0
        @test tanimoto(BitVector(), BitVector()) == 1.0
    end

    # nn_tanimoto
    @testset "nn_tanimoto" begin
        train = [
            BitVector([true, false, false]),
            BitVector([false, true, true]),
        ]
        test = [
            BitVector([true, true, false]),
            BitVector([false, false, false]),
        ]
        sims = nn_tanimoto(train, test)

        @test length(sims) == 2
        @test sims[1] ≈ tanimoto(test[1], train[1])
        @test sims[2] == 0.0
    end

    @testset "nn_distance" begin
        # Test data
        mat_train = [1.0 3.0 5.0; 2.0 4.0 6.0] # [features, samples]
        mat_test = [1.1 5.1; 2.1 6.1] # [features, samples]
        vec_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        vec_test = [[1.1, 2.1], [5.1, 6.1]]
        bit_train = [BitVector([1, 0, 1]), BitVector([0, 0, 0])]
        bit_test = [BitVector([1, 0, 0]), BitVector([0, 1, 0])]

        @testset "Matrix inputs" begin
            dists = nn_distance(mat_train, mat_test)
            @test length(dists) == numobs(mat_test)
            @test isapprox(dists[1], 0.141421, atol=1e-6)
            @test isapprox(dists[2], 0.141421, atol=1e-6)

            cos_dists = nn_distance(mat_train, mat_test; metric=CosineDist())
            @test all(0.0 .≤ cos_dists .≤ 1.0)
        end

        @testset "Vector-of-vectors inputs" begin
            # Euclidean distance
            dists = nn_distance(vec_train, vec_test)
            @test length(dists) == length(vec_test)
            @test isapprox(dists[1], 0.141421, atol=1e-6)
            @test isapprox(dists[2], 0.141421, atol=1e-6)

            # Jaccard distance for bit vectors
            jaccard_dists = nn_distance(bit_train, bit_test; metric=Jaccard())
            @test length(jaccard_dists) == 2
            @test isapprox(jaccard_dists[1], 0.5, atol=1e-6)
            @test isapprox(jaccard_dists[2], 1.0, atol=1e-6)
        end

        @testset "Edge cases" begin
            # Empty test set
            @test nn_distance(mat_train, zeros(2, 0)) == Float64[]

            # Identical points
            identical = nn_distance(mat_train, mat_train)
            @test isapprox(identical[1], 0.0, atol=1e-8)
        end

        @testset "Consistency between input types" begin
            mat_dists = nn_distance(mat_train, mat_test)
            vec_dists = nn_distance(vec_train, vec_test)
            @test mat_dists ≈ vec_dists

            mat_cos = nn_distance(mat_train, mat_test; metric=CosineDist())
            vec_cos = nn_distance(vec_train, vec_test; metric=CosineDist())
            @test mat_cos ≈ vec_cos
        end
    end

    @testset "nn_distance_batch" begin
        mat_train = [1.0 3.0 5.0; 2.0 4.0 6.0]
        mat_test  = [1.1 5.1; 2.1 6.1]

        full   = nn_distance(mat_train, mat_test)
        batch1 = nn_distance_batch(mat_train, mat_test; stride=1)
        batch2 = nn_distance_batch(mat_train, mat_test; stride=2)
        large  = nn_distance_batch(mat_train, mat_test; stride=1000)

        @test full ≈ batch1
        @test full ≈ batch2
        @test full ≈ large

        # vector-of-vectors input
        vec_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        vec_test  = [[1.1, 2.1], [5.1, 6.1]]
        @test nn_distance_batch(vec_train, vec_test; stride=1) ≈ full
    end

    @testset "tukey_hsd" begin
        # NIST example data
        values = [6.9, 5.4, 5.8, 4.6, 4.0, 8.3, 6.8, 7.8, 9.2, 6.5, 8.0, 10.5, 8.1, 6.9, 9.3, 5.8, 3.8, 6.1, 5.6, 6.2]
        groups = vcat(fill("G1", 5), fill("G2", 5), fill("G3", 5), fill("G4", 5))
        # Reference intervals from NIST
        ref_intervals = Dict(
            ("G2", "G1") => (0.29, 4.47),
            ("G3", "G1") => (1.13, 5.31),
            ("G1", "G4") => (-2.25, 1.93),
            ("G2", "G3") => (-2.93, 1.25),
            ("G2", "G4") => (0.13, 4.31),
            ("G3", "G4") => (0.97, 5.15),
        )
        results = tukey_hsd(values, groups; alpha=0.05)
        for r in results
            key = (string(r.group1), string(r.group2))
            if haskey(ref_intervals, key)
                ref_lo, ref_hi = ref_intervals[key]
                @test isapprox(r.lower, ref_lo, atol=0.05)
                @test isapprox(r.upper, ref_hi, atol=0.05)
            elseif haskey(ref_intervals, (string(r.group2), string(r.group1)))
                # Check the reverse direction
                ref_lo, ref_hi = ref_intervals[(string(r.group2), string(r.group1))]
                @test isapprox(-r.upper, ref_lo, atol=0.05)
                @test isapprox(-r.lower, ref_hi, atol=0.05)
            end
        end
        @testset "tukey_hsd_scipy" begin
            group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
            group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
            group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
            values = vcat(group0, group1, group2)
            groups = vcat(fill("G0", length(group0)), fill("G1", length(group1)), fill("G2", length(group2)))
            # Reference intervals from scipy (alpha=0.01, i.e., 99% confidence)
            ref_intervals = Dict(
                ("G0", "G1") => (-9.480, 0.280),
                ("G0", "G2") => (-5.140, 4.620),
                ("G1", "G0") => (-0.280, 9.480),
                ("G1", "G2") => (-0.540, 9.220),
                ("G2", "G0") => (-4.620, 5.140),
                ("G2", "G1") => (-9.220, 0.540),
            )
            results = tukey_hsd(values, groups; alpha=0.01)
            for r in results
                key = (string(r.group1), string(r.group2))
                if haskey(ref_intervals, key)
                    ref_lo, ref_hi = ref_intervals[key]
                    @test isapprox(r.lower, ref_lo, atol=0.05)
                    @test isapprox(r.upper, ref_hi, atol=0.05)
                end
            end
        end
    end
end
