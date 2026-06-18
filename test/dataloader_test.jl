using Test, CodecZlib, CSV, Tables, DataSplitBench

# ── helpers ──────────────────────────────────────────────────────────────────

function write_merck_csv(path::String; extra_cols=String[])
    cols = vcat(["MOLECULE", "Act", "D_1", "D_2", "D_3"], extra_cols)
    rows = [
        vcat(["M_001", 5.3, 0, 1, 0], fill(0, length(extra_cols))),
        vcat(["M_002", 4.1, 1, 0, 1], fill(0, length(extra_cols))),
        vcat(["M_003", 6.7, 0, 0, 0], fill(0, length(extra_cols))),
    ]
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
end

function write_merck_csv_gz(path_gz::String; extra_cols=String[])
    plain = path_gz[1:end-3]   # strip .gz
    write_merck_csv(plain; extra_cols)
    write(path_gz, transcode(GzipCompressor, read(plain)))
    rm(plain)
end

# ── _resolve_csv_path ─────────────────────────────────────────────────────────

@testset "resolve CSV path" begin
    dir = mktempdir()

    plain = joinpath(dir, "test.csv")
    gz    = plain * ".gz"

    write_merck_csv(plain)
    @test DataSplitBench._resolve_csv_path(plain) == plain

    mv(plain, gz)
    @test DataSplitBench._resolve_csv_path(plain) == gz   # .gz preferred

    write_merck_csv(plain)   # both present
    @test DataSplitBench._resolve_csv_path(plain) == gz   # .gz still wins

    rm(plain); rm(gz)
    @test_throws ArgumentError DataSplitBench._resolve_csv_path(plain)
end

# ── _merck_load_csv ───────────────────────────────────────────────────────────

@testset "load plain CSV" begin
    dir = mktempdir()
    path = joinpath(dir, "fake_training_disguised.csv")
    write_merck_csv(path)

    X, y, cols = DataSplitBench._merck_load_csv(path)

    @test size(X) == (3, 3)           # 3 features × 3 samples
    @test eltype(X) == Int32
    @test length(y) == 3
    @test eltype(y) == Float64
    @test y ≈ [5.3, 4.1, 6.7]
    @test cols == [:D_1, :D_2, :D_3]
end

@testset "load gzipped CSV" begin
    dir = mktempdir()
    path_gz = joinpath(dir, "fake_training_disguised.csv.gz")
    write_merck_csv_gz(path_gz)

    # pass the plain path — _resolve_csv_path should find the .gz
    plain = path_gz[1:end-3]
    X, y, cols = DataSplitBench._merck_load_csv(plain)

    @test size(X) == (3, 3)
    @test eltype(X) == Int32
    @test y ≈ [5.3, 4.1, 6.7]
    @test cols == [:D_1, :D_2, :D_3]
end

# ── merck_dataloader ──────────────────────────────────────────────────────────

@testset "merck_dataloader plain CSV" begin
    dir = mktempdir()
    write_merck_csv(joinpath(dir, "TOY_training_disguised.csv"))
    write_merck_csv(joinpath(dir, "TOY_test_disguised.csv"))

    data = DataSplitBench.merck_dataloader(dir, "TOY")

    @test size(data.train.X) == (3, 3)
    @test size(data.test.X)  == (3, 3)
    @test eltype(data.train.X) == Int32
    @test data.train.y ≈ [5.3, 4.1, 6.7]
end

@testset "merck_dataloader gzipped CSV" begin
    dir = mktempdir()
    write_merck_csv_gz(joinpath(dir, "TOY_training_disguised.csv.gz"))
    write_merck_csv_gz(joinpath(dir, "TOY_test_disguised.csv.gz"))

    data = DataSplitBench.merck_dataloader(dir, "TOY")

    @test size(data.train.X) == (3, 3)
    @test size(data.test.X)  == (3, 3)
    @test eltype(data.train.X) == Int32
    @test data.train.y ≈ [5.3, 4.1, 6.7]
end

@testset "merck_dataloader missing test column filled with zeros" begin
    dir = mktempdir()
    # training has D_1, D_2, D_3, D_4; test is missing D_4
    write_merck_csv(joinpath(dir, "TOY_training_disguised.csv"); extra_cols=["D_4"])
    write_merck_csv(joinpath(dir, "TOY_test_disguised.csv"))   # no D_4

    data = DataSplitBench.merck_dataloader(dir, "TOY")

    @test size(data.train.X, 1) == 4   # 4 features
    @test size(data.test.X,  1) == 4   # aligned to training
    @test all(data.test.X[4, :] .== 0) # missing D_4 filled with zeros
end

# ── merck_train_dataloader ────────────────────────────────────────────────────

@testset "merck_train_dataloader gzipped" begin
    dir = mktempdir()
    write_merck_csv_gz(joinpath(dir, "TOY_training_disguised.csv.gz"))

    data = DataSplitBench.merck_train_dataloader(dir, "TOY")

    @test size(data.X) == (3, 3)
    @test eltype(data.X) == Int32
    @test data.y ≈ [5.3, 4.1, 6.7]
end

@testset "merck_train_dataloader missing file throws" begin
    dir = mktempdir()
    @test_throws ArgumentError DataSplitBench.merck_train_dataloader(dir, "NONEXISTENT")
end
