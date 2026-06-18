using Test
using DataSplitBench

@testset "rdkit_fp — single SMILES" begin
    ethanol = "CCO"
    fp = rdkit_fp(ethanol)

    @test fp isa Union{Vector{Bool},Missing}
    @test !ismissing(fp)
    fp = fp::Vector{Bool}
    @test length(fp) == 2048
    @test any(fp)  # at least one bit set
end

@testset "rdkit_fp — radius and nbits kwargs" begin
    smi = "c1ccccc1"
    fp_r2 = rdkit_fp(smi; radius=2, nbits=1024)
    fp_r3 = rdkit_fp(smi; radius=3, nbits=1024)

    @test !ismissing(fp_r2) && length(fp_r2::Vector{Bool}) == 1024
    @test !ismissing(fp_r3) && length(fp_r3::Vector{Bool}) == 1024
    # different radii → different fingerprints
    @test fp_r2 != fp_r3
end

@testset "rdkit_fp — same SMILES gives same fingerprint" begin
    smi = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    @test rdkit_fp(smi) == rdkit_fp(smi)
end

@testset "rdkit_fp — invalid SMILES returns missing" begin
    fp = rdkit_fp("not_a_smiles!!!")
    @test ismissing(fp)
end

@testset "rdkit_fp — Vector{String} overload" begin
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    fps = rdkit_fp(smiles)

    @test fps isa Vector{Union{Vector{Bool},Missing}}
    @test length(fps) == 3
    @test all(!ismissing, fps)
    @test all(fp -> length(fp::Vector{Bool}) == 2048, fps)
end

@testset "rdkit_fp — Vector with invalid entry" begin
    smiles = ["CCO", "not_valid!!!", "c1ccccc1"]
    fps = rdkit_fp(smiles)

    @test !ismissing(fps[1])
    @test ismissing(fps[2])
    @test !ismissing(fps[3])
end
