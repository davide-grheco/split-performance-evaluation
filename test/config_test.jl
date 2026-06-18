using Test
using DataSplitBench

@testset "Config" begin

    # -----------------------------------------------------------------------
    # Helper: write a minimal valid TOML to a temp file
    # -----------------------------------------------------------------------
    function write_toml(content::String)
        path = tempname() * ".toml"
        write(path, content)
        return path
    end

    MINIMAL_TOML = """
    [experiment]
    name = "test_exp"
    description = "unit test"
    base_seed = 7

    [data]
    root = "Data/merck"
    cv_root = "Splits/merck"
    datasets = ["3A4", "CB1"]

    [splitting]
    methods = ["random", "kennardstone"]
    frac = 0.8
    repeats = 3
    k_folds = 4

    [models]
    methods = ["lightgbm"]
      [models.lightgbm]
      n_estimators = 100
      learning_rate = 0.1

    [output]
    root = "experiments"
    """

    @testset "load_config — minimal valid TOML" begin
        path = write_toml(MINIMAL_TOML)
        cfg = DataSplitBench.Config.load_config(path)

        @test cfg.name == "test_exp"
        @test cfg.description == "unit test"
        @test cfg.base_seed == 7

        @test cfg.data.root == "Data/merck"
        @test cfg.data.cv_root == "Splits/merck"
        @test cfg.data.datasets == ["3A4", "CB1"]
        @test cfg.data.subsample_n === nothing
        @test cfg.data.subsample_seed_offset === nothing

        @test cfg.splitting.methods == ["random", "kennardstone"]
        @test cfg.splitting.ratios == [0.8]
        @test cfg.splitting.repeats == 3
        @test cfg.splitting.k_folds == 4

        @test cfg.models.methods == ["lightgbm"]
        @test cfg.models.lightgbm.n_estimators == 100
        @test cfg.models.lightgbm.learning_rate == 0.1

        @test cfg.output.root == "experiments"
    end

    @testset "load_config — optional subsample fields" begin
        toml = MINIMAL_TOML * "\n[data]\nroot=\"D\"\ncv_root=\"S\"\ndatasets=[\"A\"]\nsubsample_n=500\nsubsample_seed_offset=99\n"
        # Use a fresh TOML that overrides [data] with subsample fields
        full = """
        [experiment]
        name = "s"
        base_seed = 1

        [data]
        root = "D"
        cv_root = "S"
        datasets = ["A"]
        subsample_n = 500
        subsample_seed_offset = 99

        [splitting]
        methods = ["random"]
        frac = 0.8
        repeats = 1
        k_folds = 2

        [models]
        methods = ["lightgbm"]
          [models.lightgbm]
          n_estimators = 10
          learning_rate = 0.01

        [output]
        root = "exp"
        """
        cfg = DataSplitBench.Config.load_config(write_toml(full))
        @test cfg.data.subsample_n == 500
        @test cfg.data.subsample_seed_offset == 99
    end

    @testset "load_config — defaults for optional experiment fields" begin
        toml = """
        [experiment]
        name = "minimal"

        [data]
        root = "D"
        cv_root = "S"
        datasets = ["X"]

        [splitting]
        methods = ["random"]

        [models]
        methods = ["lightgbm"]

        [output]
        root = "out"
        """
        cfg = DataSplitBench.Config.load_config(write_toml(toml))
        @test cfg.base_seed == 42    # default
        @test cfg.splitting.ratios == [0.8]   # default
        @test cfg.splitting.repeats == 5     # default
        @test cfg.splitting.k_folds == 5     # default
        @test cfg.models.lightgbm.n_estimators == 500   # default
        @test cfg.models.lightgbm.learning_rate == 0.05  # default
        @test cfg.models.tpe_budget == 50                 # default
        @test cfg.ad.method == "knn_distance"             # default
        @test cfg.ad.k == 5                               # default
        @test cfg.ad.z == 1.5                             # default
    end

    @testset "path helpers" begin
        path = write_toml(MINIMAL_TOML)
        cfg = DataSplitBench.Config.load_config(path)

        @test DataSplitBench.Config.experiment_root(cfg) == joinpath("experiments", "test_exp")
        @test DataSplitBench.Config.splits_root(cfg) == joinpath("experiments", "test_exp", "splits")
        @test DataSplitBench.Config.metrics_root(cfg) == joinpath("experiments", "test_exp", "metrics")
        @test DataSplitBench.Config.model_results_root(cfg) == joinpath("experiments", "test_exp", "model_results")
    end

    @testset "load_config — file not found raises" begin
        @test_throws Exception DataSplitBench.Config.load_config("/nonexistent/path/nope.toml")
    end
end
