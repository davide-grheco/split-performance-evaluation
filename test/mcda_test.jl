using Test
using DataSplitBench   # exports topsis_rank, moora_rank, mcda_rank, MCDAResult

# ---------------------------------------------------------------------------
# Independent reference implementations (vector-norm normalization), used to
# cross-check the JMcDM-backed wrappers. If two independent implementations agree
# on the scores for many matrices — including mixed benefit/cost criteria — we can
# trust both the library call and our direction handling.
# ---------------------------------------------------------------------------
function ref_topsis(X, w, benefit)
    w = w ./ sum(w)
    N = X ./ sqrt.(sum(X .^ 2; dims = 1))
    V = N .* reshape(w, 1, :)
    best  = [benefit[j] ? maximum(@view V[:, j]) : minimum(@view V[:, j]) for j in 1:size(V, 2)]
    worst = [benefit[j] ? minimum(@view V[:, j]) : maximum(@view V[:, j]) for j in 1:size(V, 2)]
    dpos = vec(sqrt.(sum((V .- reshape(best,  1, :)) .^ 2; dims = 2)))
    dneg = vec(sqrt.(sum((V .- reshape(worst, 1, :)) .^ 2; dims = 2)))
    vec(dneg ./ (dpos .+ dneg))
end

function ref_moora_ratio(X, w, benefit)
    w = w ./ sum(w)
    N = X ./ sqrt.(sum(X .^ 2; dims = 1))
    V = N .* reshape(w, 1, :)
    signs = [benefit[j] ? 1.0 : -1.0 for j in 1:size(V, 2)]
    vec(V * signs)
end

# Competition/average ranks with 1 = best (largest score first).
function ref_ranks(scores)
    order = sortperm(scores; rev = true)
    r = zeros(Float64, length(scores))
    i = 1
    while i <= length(order)
        j = i
        while j < length(order) && scores[order[j+1]] == scores[order[i]]
            j += 1
        end
        avg = (i + j) / 2      # average of the tied positions
        for k in i:j
            r[order[k]] = avg
        end
        i = j + 1
    end
    r
end

@testset "MCDA (TOPSIS / MOORA)" begin

    @testset "hand-computed tiny cases" begin
        # Two alternatives, one benefit criterion: larger value wins.
        r = topsis_rank([1.0; 2.0;;], [1.0], [true])
        @test r.scores ≈ [0.0, 1.0] atol = 1e-12
        @test r.ranks == [2.0, 1.0]

        # Same data as a cost criterion: smaller value now wins → ranking inverts.
        rc = topsis_rank([1.0; 2.0;;], [1.0], [false])
        @test rc.scores ≈ [1.0, 0.0] atol = 1e-12
        @test rc.ranks == [1.0, 2.0]

        # MOORA ratio on one benefit criterion is monotone in the value.
        m = moora_rank([1.0; 2.0;;], [1.0], [true])
        @test m.scores[2] > m.scores[1]
        @test m.ranks == [2.0, 1.0]
    end

    @testset "cross-check vs independent reference (mixed directions)" begin
        # A few fixed decision matrices with benefit AND cost criteria.
        cases = [
            ([9.0 8 7 6; 8 7 9 6; 6 9 8 7; 7 6 6 9], [0.25, 0.25, 0.25, 0.25], Bool[1, 1, 1, 1]),
            ([250.0 16 12 5; 200 16 8 3; 300 32 16 4; 275 32 8 4; 225 16 16 2],
                [0.25, 0.25, 0.25, 0.25], Bool[0, 1, 1, 1]),                     # col1 = cost
            ([1.0 5 3; 4 2 6; 7 8 1; 2 3 9], [0.5, 0.3, 0.2], Bool[1, 0, 1]),    # col2 = cost
        ]
        for (X, w, benefit) in cases
            rt = topsis_rank(X, w, benefit)
            rm = moora_rank(X, w, benefit)
            @test rt.scores ≈ ref_topsis(X, w, benefit)       rtol = 1e-8
            @test rm.scores ≈ ref_moora_ratio(X, w, benefit)  rtol = 1e-8
            @test rt.ranks == ref_ranks(ref_topsis(X, w, benefit))
            @test rm.ranks == ref_ranks(ref_moora_ratio(X, w, benefit))
            # Sanity: TOPSIS closeness always in [0, 1].
            @test all(0.0 .<= rt.scores .<= 1.0)
        end
    end

    @testset "tie handling gives average ranks" begin
        # Rows 1 and 2 identical → tied for the same score.
        X = [5.0 5.0; 5.0 5.0; 1.0 9.0]
        r = topsis_rank(X, [0.5, 0.5], [true, true])
        @test r.scores[1] ≈ r.scores[2] atol = 1e-12
        # The two tied alternatives share the average of their two rank positions.
        tied = r.ranks[1]
        @test r.ranks[1] == r.ranks[2] == tied
        @test sum(r.ranks) ≈ 1 + 2 + 3   # ranks always sum to n(n+1)/2
    end

    @testset "invariances" begin
        X = [3.0 1 4; 1 5 9; 2 6 5; 3 5 8]
        benefit = Bool[1, 0, 1]
        base = topsis_rank(X, [0.2, 0.3, 0.5], benefit)
        # Scaling all weights by a constant must not change scores or ranks.
        scaled = topsis_rank(X, [2.0, 3.0, 5.0], benefit)
        @test base.scores ≈ scaled.scores rtol = 1e-10
        @test base.ranks == scaled.ranks
        # Determinism: identical inputs → identical outputs.
        @test topsis_rank(X, [0.2, 0.3, 0.5], benefit).scores == base.scores
    end

    @testset "mcda_rank returns both methods consistently" begin
        X = [9.0 8 7 6; 8 7 9 6; 6 9 8 7; 7 6 6 9]
        w = fill(0.25, 4)
        b = Bool[1, 1, 1, 1]
        both = mcda_rank(X, w, b)
        @test both.topsis.scores == topsis_rank(X, w, b).scores
        @test both.moora.scores == moora_rank(X, w, b).scores
    end

    @testset "input validation" begin
        X = [1.0 2; 3 4]
        @test_throws DimensionMismatch topsis_rank(X, [1.0], [true, true])     # wrong weights length
        @test_throws DimensionMismatch topsis_rank(X, [1.0, 1.0], [true])      # wrong benefit length
        @test_throws ArgumentError     topsis_rank(X, [0.0, 0.0], [true, true])# non-positive weights
        @test_throws DimensionMismatch moora_rank(X, [1.0], [true, true])
    end
end
