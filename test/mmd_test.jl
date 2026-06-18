using Test
using Random
using RDatasets
using KernelFunctions


@testset "MMD Tests" begin

    Random.seed!(42)

    @testset "Identical distributions" begin
        X = randn(2, 100)
        Y = copy(X)
        result = mmd(X, Y)
        @test result ≈ 0 atol = 1e-6
    end

    @testset "Different distributions" begin
        X = randn(2, 100)
        Y = randn(2, 100) .+ 1.0
        result = mmd(X, Y)
        @test result > 0
    end

    @testset "Different sized inputs" begin
        X = randn(2, 100)
        Y = randn(2, 80)
        result = mmd(X, Y)
        @test result > 0
        X = randn(2, 80)
        Y = randn(2, 100)
        result = mmd(X, Y)
        @test result > 0
    end

    @testset "Small Matrices" begin
        X = [1.0 3.0; 2.0 4.0]
        Y = [1.5 3.5; 2.5 4.5]

        expected = 52 / 4 + 74 / 4 - 2 * 62 / 4

        result = mmd(X, Y, kernel=LinearKernel())
        @test result ≈ expected atol = 1e-6

        X = [1 4 7; 2 5 8; 3 6 9]
        Y = [7 4 1 0; 6 3 1 2; 5 2 8 5]

        @test mmd(X, Y, kernel=LinearKernel()) ≈ 6 atol = 1e-6
        @test mmd(X, Y, kernel=SqExponentialKernel()) ≈ 0.56878 atol = 1e-5
        @test mmd(X, Y, kernel=PolynomialKernel()) ≈ 2436.5 atol = 1e-6

        X = [0.0 1.0; 0.0 1.0]
        Y = [1.0 0.0; 0.0 1.0]

        expected = 0.154818121

        result = mmd(X, Y, kernel=SqExponentialKernel())
        @test result ≈ expected atol = 1e-6
    end


    @testset "Empty inputs" begin
        X = zeros(2, 0)
        Y = randn(2, 10)
        @test_throws BoundsError mmd(X, Y)
    end

    @testset "Iris Setosa vs Virginica (regression test)" begin
        iris = dataset("datasets", "iris")
        setosa = Matrix(iris[iris.:Species.=="setosa", 1:4])' |> float
        virginica = Matrix(iris[iris.:Species.=="virginica", 1:4])' |> float

        @test isapprox(mmd(setosa, virginica), 1.3210144686317615; atol=1e-3)
    end
end
