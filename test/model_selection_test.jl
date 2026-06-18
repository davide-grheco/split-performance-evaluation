using Test
using Random
using DataSplitBench
using MLJ

@testset "ModelSelection" begin

    rng = MersenneTwister(42)
    X   = MLJ.table(randn(rng, 60, 4))
    y   = randn(rng, 60)

    @testset "tpe_search — unknown model throws" begin
        @test_throws ErrorException tpe_search("bogus", X, y)
        @test_throws ErrorException tpe_search(:bogus,  X, y)
    end

    @testset "tpe_search — String and Symbol give same type" begin
        p1, r1, m1 = tpe_search("random_forest", X, y; budget=5, n_inner_folds=2, seed=1)
        p2, r2, m2 = tpe_search(:random_forest,  X, y; budget=5, n_inner_folds=2, seed=1)
        @test typeof(p1) == typeof(p2)
        @test typeof(r1) == typeof(r2)
        @test typeof(m1) == typeof(m2)
    end

    @testset "tpe_search — random_forest integration" begin
        best_params, best_cv_rms, mach = tpe_search(
            :random_forest, X, y; budget=12, n_inner_folds=2, seed=42
        )

        @test best_params  isa Dict{String,Any}
        @test best_cv_rms  isa Float64
        @test best_cv_rms  > 0.0

        @test haskey(best_params, "n_trees")
        @test haskey(best_params, "max_depth")
        @test haskey(best_params, "min_samples_leaf")
        @test best_params["n_trees"]          isa Int
        @test best_params["max_depth"]        isa Int
        @test best_params["min_samples_leaf"] isa Int
        @test best_params["n_trees"]          in 50:50:500
        @test best_params["max_depth"]        in [-1, 5, 10, 20]
        @test best_params["min_samples_leaf"] in 1:20

        ŷ = MLJ.predict(mach, X)
        @test length(ŷ) == 60
        @test all(isfinite, ŷ)
    end

    @testset "tpe_search — lightgbm integration" begin
        best_params, best_cv_rms, mach = tpe_search(
            :lightgbm, X, y; budget=12, n_inner_folds=2, seed=7
        )

        @test best_params  isa Dict{String,Any}
        @test best_cv_rms  isa Float64
        @test best_cv_rms  > 0.0

        @test haskey(best_params, "num_iterations")
        @test haskey(best_params, "learning_rate")
        @test haskey(best_params, "num_leaves")
        @test haskey(best_params, "feature_fraction")
        @test best_params["num_iterations"]   isa Int
        @test best_params["num_iterations"]   in 100:100:500
        @test best_params["num_leaves"]       isa Int
        @test 15 ≤ best_params["num_leaves"]  ≤ 255
        @test 0.005 ≤ best_params["learning_rate"] ≤ 0.3
        @test 0.5   ≤ best_params["feature_fraction"] ≤ 1.0

        ŷ = MLJ.predict(mach, X)
        @test length(ŷ) == 60
        @test all(isfinite, ŷ)
    end

    @testset "tpe_search — knn integration" begin
        best_params, best_cv_rms, mach = tpe_search(
            :knn, X, y; budget=10, n_inner_folds=2, seed=3
        )

        @test best_params isa Dict{String,Any}
        @test best_cv_rms > 0.0
        @test haskey(best_params, "K")
        @test haskey(best_params, "weights")
        @test haskey(best_params, "metric")
        @test best_params["K"]       isa Int
        @test 1 ≤ best_params["K"]   ≤ 25
        @test best_params["weights"] in ["uniform", "inverse"]
        @test best_params["metric"]  in ["euclidean", "cityblock"]

        ŷ = MLJ.predict(mach, X)
        @test length(ŷ) == 60
        @test all(isfinite, ŷ)
    end

    @testset "tpe_search — svr integration" begin
        best_params, best_cv_rms, mach = tpe_search(
            :svr, X, y; budget=10, n_inner_folds=2, seed=5
        )

        @test best_params isa Dict{String,Any}
        @test best_cv_rms > 0.0
        @test haskey(best_params, "cost")
        @test haskey(best_params, "gamma")
        @test haskey(best_params, "epsilon")
        @test 0.01   ≤ best_params["cost"]    ≤ 100.0
        @test 1e-4   ≤ best_params["gamma"]   ≤ 10.0
        @test 1e-3   ≤ best_params["epsilon"] ≤ 1.0

        ŷ = MLJ.predict(mach, X)
        @test length(ŷ) == 60
        @test all(isfinite, ŷ)
    end

    @testset "tpe_search — elasticnet integration" begin
        best_params, best_cv_rms, mach = tpe_search(
            :elasticnet, X, y; budget=10, n_inner_folds=2, seed=9
        )

        @test best_params isa Dict{String,Any}
        @test best_cv_rms > 0.0
        @test haskey(best_params, "lambda")
        @test haskey(best_params, "gamma")
        @test 1e-4 ≤ best_params["lambda"] ≤ 10.0
        @test 0.0  ≤ best_params["gamma"]  ≤ 1.0

        ŷ = MLJ.predict(mach, X)
        @test length(ŷ) == 60
        @test all(isfinite, ŷ)
    end

    @testset "select_champion — returns one result per candidate" begin
        model_names = ["random_forest", "knn", "elasticnet"]
        results = select_champion(model_names, X, y;
                                  budget=8, n_inner_folds=2, base_seed=42)

        @test results isa Vector{CandidateResult}
        @test length(results) == 3

        # Exactly one champion
        champions = filter(r -> r.is_champion, results)
        @test length(champions) == 1

        # All results have valid types and can predict
        for r in results
            @test r.name          isa String
            @test r.params        isa Dict{String,Any}
            @test r.cv_rms        isa Float64
            @test r.cv_rms        > 0.0
            @test r.is_champion   isa Bool
            @test r.round_eliminated isa Int
            @test r.round_eliminated in (0, 1, 2)

            mach = retrain_candidate(r, X, y, 42)
            ŷ    = MLJ.predict(mach, X)
            @test length(ŷ) == 60
            @test all(isfinite, ŷ)
        end

        # Champion has round_eliminated == 0
        champion = champions[1]
        @test champion.round_eliminated == 0

        # Non-champions have round_eliminated > 0
        for r in filter(r -> !r.is_champion, results)
            @test r.round_eliminated > 0
        end

        # Champion has the lowest cv_rms among all results
        @test champion.cv_rms == minimum(r.cv_rms for r in results)
    end

end
