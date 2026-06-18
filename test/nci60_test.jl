using Dates

@testset "NCI-60 Date" begin

    @test DataSplitBench.nci60_date("1102NS11") == Date(2011, 02, 01)
    @test DataSplitBench.nci60_date("0004NS35") == Date(2000, 04, 01)

end
