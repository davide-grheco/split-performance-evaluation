# Friedman.jl
# Friedman two-way ANOVA by ranks
#
# Copyright (C) 2012   Simon Kornblith
# Copyright (C) 2026   Davide Crucitti
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

using HypothesisTests
using StatsAPI
using Distributions
using StatsBase

export FriedmanTest, iman_davenport

struct FriedmanTest{T<:Real} <: HypothesisTests.HypothesisTest
    n::Int                    # number of observations
    k::Int                    # number of treatments
    df::Int                   # degrees of freedom
    rank_sums::Vector{T}      # rank sums vector
    Q::T                      # Q statistic
end

"""
    FriedmanTest(groups::AbstractVector{<:Real}...)

Perform the Friedman two-way ANOVA by ranks, a rank sum test to test the difference of ``k``
treatments across ``N`` repeated tests. This is a non-parametric test similar to the
Kruskall-Wallis one-way ANOVA by ranks. It is a special case of the Durbin test.

The p-value is computed using a ``Q`` statistic, as follows:
```math
    Q = \\frac{12}{kN(k+1)} \\sum_{j = 1}^{k}(R_j^2) - 3n(k+1)
```
where ``N`` is the number of tests, ``k`` is the number of treatments, and ``R`` is the rank
sum vector.

Implements: [`pvalue`](@ref)

# References

  * Daniel, W.W., Friedman two-way analysis of variance by ranks. Applied Nonparametric Statistics
    (2nd ed.). Boston: PWS-Kent. pp. 262–74, 1990.

# External links

  * [Friedman test on Wikipedia
    ](https://en.wikipedia.org/wiki/Friedman_test)
"""
function FriedmanTest(groups::AbstractVector{T}...) where {T<:Real}
    length(groups) >= 2 || throw(ArgumentError("FriedmanTest requires at least two groups"))
    n = length(first(groups))
    n > 0 || throw(ArgumentError("groups must be non-empty"))
    all(g -> length(g) == n, groups) || throw(DimensionMismatch("all groups must have the same length"))

    x = hcat(groups...)
    k = size(x, 2)
    df = k - 1
    R = convert_rank(x)
    rank_sums = vec(sum(R; dims=1))
    tie_correction = get_tie_correction(x)
    Q = ((12 / (k * n * (k + 1))) * sum(abs2, rank_sums) - 3n * (k + 1)) / tie_correction
    FriedmanTest(n, k, df, rank_sums, Q)
end

HypothesisTests.testname(::FriedmanTest) = "Friedman two-way ANOVA by ranks"
HypothesisTests.population_param_of_interest(::FriedmanTest) = ("location parameter", "all equal", NaN) # parameter of interest: name, value under h0, point estimate
HypothesisTests.default_tail(::FriedmanTest) = :right

function HypothesisTests.show_params(io::IO, x::FriedmanTest, ident)
    println(io, ident, "number of observations: ", x.n)
    println(io, ident, "number of treatments:   ", x.k)
    println(io, ident, "degrees of freedom:     ", x.df)
    println(io, ident, "rank sums vector:       ", x.rank_sums)
    println(io, ident, "Q statistic:            ", x.Q)
end

StatsAPI.pvalue(x::FriedmanTest; tail=:right) = pvalue(Chisq(x.df), x.Q, tail=tail)

# helper functions

function convert_rank(x::AbstractMatrix{<:Real})
    R = Matrix{Float64}(undef, size(x))
    for (i, row) in enumerate(eachrow(x))
        @views R[i, :] .= tiedrank(row)
    end
    return R
end

function row_tie_count(row::AbstractVector{<:Real})
    counts = values(countmap(row))
    sum(t -> t^3 - t, counts)
end


function get_tie_correction(x::AbstractMatrix{<:Real})
    n, k = size(x)
    row_ties = sum(row_tie_count, eachrow(x))
    tc = 1 - row_ties / (n * k * (k^2 - 1))
    # Clamp to avoid Q = NaN/Inf when all values in every row are identical
    return max(tc, 1e-10)
end


function iman_davenport(ft::FriedmanTest)
    n, k, Q = ft.n, ft.k, ft.Q
    den = n * (k - 1) - Q
    den > 0 || throw(DomainError(den, "Iman-Davenport statistic undefined because denominator is nonpositive"))
    F = ((n - 1) * Q) / den
    p = 1 - cdf(FDist(k - 1, (k - 1) * (n - 1)), F)
    return F, p
end
