using Dates


GI50HEADERS = ["RELEASE_DATE",
    "EXPID",
    "PREFIX",
    "NSC",
    "CONCENTRATION_UNIT",
    "LOG_HI_CONCENTRATION",
    "PANEL_NUMBER",
    "CELL_NUMBER",
    "CELL_NAME",
    "PANEL_CODE",
    "COUNT",
    "AVERAGE",
    "STDDEV"
]

"""
    Previous data releases reported aggregate values across experiments, grouping the data by NSC number and the log of the highest concentration, rounded to one decimal point, in the concentration/response dilution series. Now, we report data for individual experiments identified by an EXPID, and all values are reported to 4 decimal places.
    All cell lines for an individual EXPID are grown and assayed contemporaneously.
    The format of the EXPID is YYMMLLSS, where YY is the last 2 digits of the year (00 - 21 for 2000 to 2021, MM is the month number (01 for January - 12 for December), LL is a pair of letters for internal process tracking and SS is a 2-digit numeric sequence.
    There are 60 cell lines in the current NCI60 cell line screen. There are 11 other cell lines that were part of the NCI60 screen in the past. These 71 cell lines comprise most of the public data. This data release also includes other cell lines which have been assayed at least once using the same protocols as the NCI60 cell line screen.
    Experimental QC checks were performed at the lab-level and during data-processing at the time that the experiments were run. Additional quality control or consistency checks have not been performed.
    Endpoint values (GI50, TGI, LC50) have accompanying concentration/response data; however, not all concentration/response data have accompanying endpoint values.
    Most NSC numbers represent small molecules, and the reported concentrations use a "M" (molar) CONCENTRATION_UNIT.  For more complex biological agents concentrations may be reported as µg/ml (micrograms per milliliter) with a CONCENTRATION_UNIT "u".  Mixtures, extracts, crude fractions, etc.  in the assay may use units of µg/ml or volume-based measurements designated by CONCENTRATION_UNIT "V". (There is no further definition regarding what a volume-based concentration means.)
    At the level of individual experiments almost all endpoint values will have a count of 1 and a standard deviation of 0. In a few cases, though, multiple replicate determinations were run within a single experiment.
    Most of the concentration/response data are from a series of 5 dilutions at log intervals (10-fold dilution); however, a few experiments were run with 10 dilutions at half-log intervals for some cell lines and NSC compounds.
    PTC is the Percent of Treated cell growth as a fraction of Control cell growth. The IC50 endpoint values are interpolated from these data.
    GIPRCNT is the percent of treated cell growth as a fraction of control cell growth corrected for the count of cells at the time of drug addition in the assay. 100 is control growth, 0 is complete inhibition of growth (cytostasis), and -100 is complete cell kill. The GI50, TGI and LC50 values are determined by simple interpolation of GIPRCNT values above and below 50, 0, and -50 respectively.
    Where GI50, TGI or LC50 values would be outside the concentration range of the dilution series the highest or lowest concentration in the series is reported.
    For historical reasons, PTC values were not stored in our database. They have been recalculated for this data release. There are a few cases where we report no PTC value but do report a GIPRCNT value. The data processing is able to work with certain occurrences of null values within the series of concentration/response data.
    The ONECONC prescreen has changed over the years. Currently all 60 of the NCI60 cell lines are evaluated, but in the past the assay was run against only a small number of cell lines.

    GI50, TGI, LC50, IC50 VALUES

    RELEASE_DATE The date of this data release.
    EXPID Please see the General Comments above.
    PREFIX The identifier of the sequence from which an NSC number was assigned. All public data are in the S series.
    NSC The numeric identifier in the S series.
    CONCENTRATION_UNIT Please see the General Comments above.
    LOG_HI_CONCENTRATION The log10 of the highest concentration of the concentration/response data.
    PANEL_NUMBER Internal identifier. The combinations of panel_number and cell_number are unique cell line identifiers.
    CELL_NUMBER Internal identifier. The combinations of panel_number and cell_number are unique cell line identifiers.
    PANEL_NAME The name of the NCI cell line panel (cancer type).
    CELL_NAME The name of the NCI cell line.
    PANEL_CODE An abbreviation for the panel_name.
    COUNT Count of interpolated values.
    AVERAGE Average of interpolated values.
    STDDEV Standard deviation of interpolated values.
"""
function nci60_load_all(activity_csv::String, smiles_csv::String)
    # read
    act = CSV.read(activity_csv, DataFrame)
    smi = CSV.read(smiles_csv, DataFrame)

    # basic checks
    for c in (:NSC, :CELL_NAME, :AVERAGE, :EXPID)
        @assert c in names(act) "Missing column $c in activity CSV"
    end
    @assert :SMILES in names(smi) "Missing column :SMILES in SMILES CSV"
    @assert :NSC in names(smi) "Missing column :NSC in SMILES CSV"

    # normalize types for join
    act[!, :NSC] = string.(act.NSC)
    smi[!, :NSC] = string.(smi.NSC)
    smi = unique(smi, :NSC)

    # choose latest RELEASE_DATE per (NSC, CELL_NAME)
    cell_line = string(panel_number, "_", cell_number)
    act_dedup = unique(act, [:NSC, :PANEL_NAME])


    # join SMILES
    df = innerjoin(act_dedup, smi, on=:NSC)

    # rename AVERAGE -> NLOGGI50_N (the −log10(GI50) used by your other code)
    rename!(df, :AVERAGE => :NLOGGI50_N)

    # tidy up: drop rows with missing SMILES or target
    df = dropmissing(df, [:SMILES, :NLOGGI50_N])

    return df
end

"""
Interpret the NCI-60 EXPID into the date of the experiment
The format of the EXPID is YYMMLLSS, where YY is the last 2 digits of the year (00 - 21 for 2000 to 2021, MM is the month number (01 for January - 12 for December),
LL is a pair of letters for internal process tracking and SS is a 2-digit numeric sequence.
"""
function nci60_date(expid::AbstractString)::Date
    expid = strip(expid)
    @assert length(expid) == 8

    yy = parse(Int, expid[1:2])
    mm = parse(Int, expid[3:4])

    year = 2000 + yy

    return Date(year, mm, 1)
end


CURRENT_CELL_LINES = [
    "MCF7",
    "MDA-MB-231/ATCC",
    "HS 578T",
    "BT-549",
    "T-47D",
    "SF-268",
    "SF-295",
    "SF-539",
    "SNB-19",
    "SNB-75",
    "U251",
    "COLO 205",
    "HCC-2998",
    "HCT-116",
    "HCT-15",
    "HT29",
    "KM12",
    "SW-620",
    "CCRF-CEM",
    "HL-60(TB)",
    "K-562",
    "MOLT-4",
    "RPMI-8226",
    "SR",
    "LOX IMVI",
    "MALME-3M",
    "M14",
    "SK-MEL-2",
    "SK-MEL-28",
    "SK-MEL-5",
    "UACC-257",
    "UACC-62",
    "MDA-MB-435",
    "MDA-N",
    "A549/ATCC",
    "EKVX",
    "HOP-62",
    "HOP-92",
    "NCI-H226",
    "NCI-H23",
    "NCI-H322M",
    "NCI-H460",
    "NCI-H522",
    "IGROV1",
    "OVCAR-3",
    "OVCAR-4",
    "OVCAR-5",
    "OVCAR-8",
    "SK-OV-3",
    "NCI/ADR-RES",
    "PC-3",
    "DU-145",
    "786-0",
    "A498",
    "ACHN",
    "CAKI-1",
    "RXF 393",
    "SN12C",
    "TK-10",
    "UO-31",
]
