using Test
using Statistics
using DataSplitBench

@testset "ApplicabilityDomain" begin

    # Shared data: 3 features × 5 samples
    X_train = Float64[
        1.0 2.0 3.0 4.0 5.0;
        2.0 1.0 4.0 3.0 5.0;
        5.0 3.0 1.0 4.0 2.0
    ]

    # X_in is close in direction to the training cloud; X_out points in the opposite
    # direction (negative orthant) which is far from all training vectors under CosineDist.
    X_in = reshape([3.0, 3.0, 3.0], 3, 1)
    X_out = reshape([-1.0, -1.0, -1.0], 3, 1)
    X_test = hcat(X_in, X_out)

    @testset "BoundingBoxAD" begin
        ad = fit_ad(BoundingBoxAD, X_train)
        @test ad.feature_min == [1.0, 1.0, 1.0]
        @test ad.feature_max == [5.0, 5.0, 5.0]

        scores = ad_score(ad, X_test)
        @test scores[1] == 0.0        # inside
        @test scores[2] == 1.0        # fully outside

        mask = in_domain(ad, X_test)
        @test mask[1] == true
        @test mask[2] == false

        # All training points should be inside their own bounding box
        @test all(in_domain(ad, X_train))
    end

    @testset "KNNDistanceAD" begin
        ad = fit_ad(KNNDistanceAD, X_train; k=2, z=3.0)
        @test ad.k == 2
        @test ad.threshold > 0.0

        scores = ad_score(ad, X_test)
        @test scores[1] < scores[2]

        mask = in_domain(ad, X_test)
        @test mask[1] == true
        @test mask[2] == false
    end

    @testset "LeverageAD" begin
        ad = fit_ad(LeverageAD, X_train)
        p, n = size(X_train)
        @test ad.threshold ≈ 3(p + 1) / n

        scores = ad_score(ad, X_test)
        @test scores[1] < scores[2]

        mask = in_domain(ad, X_test)
        @test mask[1] == true
        @test mask[2] == false

        # hat values must be non-negative
        @test all(>=(0.0), ad_score(ad, X_train))
    end

    @testset "TanimotoAD" begin
        train_fps = [
            BitVector([1, 0, 1, 0]),
            BitVector([1, 1, 0, 0]),
            BitVector([0, 1, 1, 0]),
        ]
        fp_in = BitVector([1, 0, 1, 0])   # identical to train[1]
        fp_out = BitVector([0, 0, 0, 1])   # no overlap with any training fp

        ad = fit_ad(TanimotoAD, train_fps; z=1.0)
        @test 0.0 <= ad.threshold <= 1.0

        scores = ad_score(ad, [fp_in, fp_out])
        @test scores[1] > scores[2]

        mask = in_domain(ad, [fp_in, fp_out])
        @test mask[1] == true
        @test mask[2] == false
    end
end
