using Test

@testset "Small matrix sanity test for mahalanobis_split_distance" begin
    train = [1.0 3.0;
        3.0 3.0]

    test = [4.0 4.0;
        4.0 6.0]

    Λ = mahalanobis_split_distance(train, test)

    @test isa(Λ, Float64)
    @test Λ > 0.0
    @test isapprox(Λ, 9, atol=1e-5)
end
