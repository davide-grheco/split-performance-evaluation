using Test
using DataFrames
using DataSplitBench
using Statistics

# ---------------------------------------------------------------------------
# Mirror the logic from scripts/run_ad_annotation.jl for unit testing.
# ---------------------------------------------------------------------------

function normalise_shard(df::DataFrame)
    if :is_champion in propertynames(df)
        df[:, [:sample_id, :model, :is_champion, :y_true, :y_pred]]
    else
        agg = combine(groupby(df, [:sample_id, :model]),
            :y_true => first => :y_true,
            :y_pred => mean  => :y_pred,
        )
        agg[!, :is_champion] .= true
        agg[:, [:sample_id, :model, :is_champion, :y_true, :y_pred]]
    end
end

function make_ad_annotation_shard(dataset, splitter, ratio, rep, fold, set,
                                   sample_ids, ad_scores, in_domain_flags, nn_dists,
                                   pred_df::DataFrame)
    ad_df = DataFrame(
        sample_id        = collect(Int, sample_ids),
        ad_score         = collect(Float64, ad_scores),
        in_domain        = collect(Bool, in_domain_flags),
        nn_dist_to_train = collect(Float64, nn_dists),
    )
    joined = leftjoin(pred_df, ad_df; on=:sample_id)
    n = nrow(joined)
    hcat(DataFrame(
        dataset  = fill(dataset,  n),
        splitter = fill(splitter, n),
        ratio    = fill(ratio,    n),
        rep      = fill(rep,      n),
        fold     = fill(fold,     n),
        set      = fill(set,      n),
    ), joined)
end

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Model-selection shard: one row per (sample_id, model_candidate)
function mock_model_selection_shard(sample_ids, models; champion_idx=1)
    n_samples = length(sample_ids)
    n_models  = length(models)
    DataFrame(
        sample_id   = repeat(sample_ids,  inner=n_models),
        model       = repeat(models,      outer=n_samples),
        is_champion = repeat(
            [i == champion_idx for i in 1:n_models], outer=n_samples
        ),
        y_true      = repeat(Float64.(sample_ids), inner=n_models),
        y_pred      = Float64.(repeat(1:n_models, outer=n_samples)) .* 0.1 .+
                      repeat(Float64.(sample_ids), inner=n_models),
    )
end

# Layer 1 shard: one row per (sample_id, run), no is_champion
function mock_layer1_shard(sample_ids, model, n_runs)
    n = length(sample_ids)
    DataFrame(
        sample_id = repeat(sample_ids, inner=n_runs),
        model     = fill(model, n * n_runs),
        run       = repeat(1:n_runs, outer=n),
        y_true    = repeat(Float64.(sample_ids), inner=n_runs),
        y_pred    = repeat(Float64.(sample_ids), inner=n_runs) .+
                    repeat(range(-0.5, 0.5; length=n_runs), outer=n),
    )
end

# ---------------------------------------------------------------------------
# normalise_shard
# ---------------------------------------------------------------------------

@testset "normalise_shard" begin

    @testset "model-selection path: keeps per-model rows, preserves is_champion" begin
        df = mock_model_selection_shard([1, 2], ["m1", "m2", "m3"]; champion_idx=2)
        out = normalise_shard(df)
        @test nrow(out) == 6           # 2 samples × 3 models
        @test :is_champion in propertynames(out)
        @test :model in propertynames(out)
        @test all(filter(r -> r.model == "m2", out).is_champion .== true)
        @test all(filter(r -> r.model != "m2", out).is_champion .== false)
        # y_pred unchanged (no averaging)
        for row in eachrow(filter(r -> r.model == "m1", out))
            @test row.y_pred ≈ 0.1 + row.sample_id
        end
    end

    @testset "layer1 path: averages over runs, adds is_champion=true" begin
        df = mock_layer1_shard([10, 20], "lightgbm", 10)
        out = normalise_shard(df)
        @test nrow(out) == 2           # one row per sample after averaging
        @test :is_champion in propertynames(out)
        @test all(out.is_champion .== true)
        @test all(out.model .== "lightgbm")
        # y_pred should be the mean of the 10 runs (which are centred on y_true)
        @test out.y_pred ≈ out.y_true  atol=1e-10
    end

    @testset "output schema identical for both paths" begin
        ms  = normalise_shard(mock_model_selection_shard([1], ["a", "b"]))
        l1  = normalise_shard(mock_layer1_shard([1], "a", 5))
        @test propertynames(ms) == propertynames(l1)
    end
end

# ---------------------------------------------------------------------------
# make_ad_annotation_shard — model-selection path
# ---------------------------------------------------------------------------

@testset "AD annotation shard — model-selection path" begin

    sample_ids = [10, 20, 30]
    models     = ["lightgbm", "random_forest", "knn"]
    pred_df    = normalise_shard(mock_model_selection_shard(sample_ids, models; champion_idx=1))

    ad_scores       = [0.5, 1.2, 4.8]
    in_domain_flags = [true, true, false]
    nn_dists        = [1.0, 2.0, 8.0]

    result = make_ad_annotation_shard(
        "TESTDS", "random", 0.8, 1, 2, "test",
        sample_ids, ad_scores, in_domain_flags, nn_dists, pred_df,
    )

    required = [:dataset, :splitter, :ratio, :rep, :fold, :set,
                :sample_id, :model, :is_champion, :y_true, :y_pred,
                :ad_score, :in_domain, :nn_dist_to_train]

    @testset "schema" begin
        for col in required
            @test col in propertynames(result)
        end
    end

    @testset "row count — one row per (sample, model)" begin
        @test nrow(result) == length(sample_ids) * length(models)
    end

    @testset "AD scores broadcast across models for each sample" begin
        for (sid, expected_score, expected_domain, expected_nn) in
                zip(sample_ids, ad_scores, in_domain_flags, nn_dists)
            rows = filter(r -> r.sample_id == sid, result)
            @test nrow(rows) == length(models)
            @test all(rows.ad_score         .≈ expected_score)
            @test all(rows.in_domain        .== expected_domain)
            @test all(rows.nn_dist_to_train .≈ expected_nn)
        end
    end

    @testset "per-model y_pred preserved, not averaged" begin
        for (mi, m) in enumerate(models)
            rows = filter(r -> r.model == m, result)
            @test nrow(rows) == length(sample_ids)
            @test rows.y_pred ≈ mi * 0.1 .+ Float64.(sample_ids)
        end
    end

    @testset "is_champion preserved" begin
        @test all(filter(r -> r.model == "lightgbm",      result).is_champion .== true)
        @test all(filter(r -> r.model != "lightgbm",      result).is_champion .== false)
    end

    @testset "stratification columns" begin
        @test all(result.dataset  .== "TESTDS")
        @test all(result.splitter .== "random")
        @test all(result.ratio    .≈ 0.8)
        @test all(result.rep      .== 1)
        @test all(result.fold     .== 2)
        @test all(result.set      .== "test")
    end

    @testset "IPS UnitRange sample_ids" begin
        n_ips   = 4
        ips_df  = normalise_shard(mock_model_selection_shard(collect(1:n_ips), models))
        ips_res = make_ad_annotation_shard(
            "TESTDS", "random", 0.8, 1, 1, "ips",
            1:n_ips, rand(n_ips), fill(true, n_ips), rand(n_ips), ips_df,
        )
        @test nrow(ips_res) == n_ips * length(models)
        @test all(ips_res.set .== "ips")
    end
end

# ---------------------------------------------------------------------------
# make_ad_annotation_shard — layer 1 path
# ---------------------------------------------------------------------------

@testset "AD annotation shard — layer 1 path" begin

    sample_ids = [5, 10]
    l1_df = normalise_shard(mock_layer1_shard(sample_ids, "lightgbm", 10))

    result = make_ad_annotation_shard(
        "D", "butina", 0.8, 2, 3, "test",
        sample_ids, [0.3, 2.1], [true, false], [0.5, 3.0], l1_df,
    )

    @testset "one row per sample (not per run)" begin
        @test nrow(result) == length(sample_ids)
    end

    @testset "is_champion true for sole model" begin
        @test all(result.is_champion .== true)
    end

    @testset "y_pred is the mean over runs" begin
        @test result.y_pred ≈ result.y_true  atol=1e-10
    end

    @testset "AD scores present" begin
        @test result.ad_score       ≈ [0.3, 2.1]
        @test result.in_domain      == [true, false]
        @test result.nn_dist_to_train ≈ [0.5, 3.0]
    end
end

# ---------------------------------------------------------------------------
# Regression: no cross-model averaging in model-selection path
# ---------------------------------------------------------------------------

@testset "no-averaging regression" begin
    df = DataFrame(
        sample_id   = [1, 1],
        model       = ["m1", "m2"],
        is_champion = [true, false],
        y_true      = [3.0, 3.0],
        y_pred      = [1.0, 9.0],
    )
    pred = normalise_shard(df)
    result = make_ad_annotation_shard(
        "D", "s", 0.8, 1, 1, "test",
        [1], [0.1], [true], [0.5], pred,
    )
    m1 = only(filter(r -> r.model == "m1", result).y_pred)
    m2 = only(filter(r -> r.model == "m2", result).y_pred)
    @test m1 ≈ 1.0
    @test m2 ≈ 9.0
    @test m1 != m2
end
