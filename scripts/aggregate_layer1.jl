#!/usr/bin/env julia
#
# aggregate_layer1.jl — compute per-(dataset, splitter, ratio, model, rep, fold)
# summary metrics from Layer 1 raw prediction shards and write a summary Arrow.
#
# For each (splitter, ratio, rep, fold), metrics are computed separately for the
# internal test set ("test") and the external benchmark ("ips"), then pivoted so
# each output row has both sets' metrics plus signed/absolute gaps:
#
#   rmse_gap      = rmse_test  - rmse_ips   (positive → internal looks harder)
#   r2_gap        = r2_test    - r2_ips     (positive → internal looks better)
#   abs_rmse_gap  = |rmse_gap|              (absolute estimation error)
#   bias_gap      = bias_test  - bias_ips   (signed difference in mean residuals)
#
# Usage (called by Snakemake):
#   # Preferred: pass shard paths via a file to avoid shell arg-length limits:
#   julia --project=. scripts/aggregate_layer1.jl <out_arrow> @<shards_list_file>
#
#   # Legacy: positional args still work for small shard counts:
#   julia --project=. scripts/aggregate_layer1.jl <out_arrow> <shard1> [shard2 ...]

using Arrow
using DataFrames
using Statistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function compute_metrics(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    n = length(y_true)
    n == 0 && return (rmse=NaN, r2=NaN, mae=NaN, bias=NaN)
    resid   = y_pred .- y_true
    ss_res  = sum(resid .^ 2)
    ss_tot  = sum((y_true .- mean(y_true)) .^ 2)
    rmse    = sqrt(ss_res / n)
    r2      = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    mae     = mean(abs.(resid))
    bias    = mean(resid)
    return (rmse=rmse, r2=r2, mae=mae, bias=bias)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main(out_path, shard_paths)
    isempty(shard_paths) && error("No shard paths provided")

    @info "aggregate_layer1: loading $(length(shard_paths)) shards"

    df = mapreduce(vcat, shard_paths) do p
        DataFrame(Arrow.Table(p))
    end

    @info "Total rows loaded" rows=nrow(df)

    group_keys = [:dataset, :splitter, :ratio, :model, :rep, :fold, :set]
    agg = combine(groupby(df, group_keys)) do sub
        m = compute_metrics(sub.y_true, sub.y_pred)
        DataFrame(rmse=m.rmse, r2=m.r2, mae=m.mae, bias=m.bias)
    end

    # Pivot: one row per (dataset, splitter, ratio, model, rep, fold)
    test_df = filter(r -> r.set == "test", agg)
    ips_df  = filter(r -> r.set == "ips",  agg)

    id_keys = [:dataset, :splitter, :ratio, :model, :rep, :fold]

    rename!(test_df, :rmse => :rmse_test, :r2 => :r2_test, :mae => :mae_test, :bias => :bias_test)
    rename!(ips_df,  :rmse => :rmse_ips,  :r2 => :r2_ips,  :mae => :mae_ips,  :bias => :bias_ips)

    summary = innerjoin(
        select(test_df, [id_keys; [:rmse_test, :r2_test, :mae_test, :bias_test]]...),
        select(ips_df,  [id_keys; [:rmse_ips,  :r2_ips,  :mae_ips,  :bias_ips]]...);
        on=id_keys,
    )

    summary.rmse_gap     = summary.rmse_test .- summary.rmse_ips
    summary.r2_gap       = summary.r2_test   .- summary.r2_ips
    summary.abs_rmse_gap = abs.(summary.rmse_gap)
    summary.bias_gap     = summary.bias_test .- summary.bias_ips

    mkpath(dirname(out_path))
    Arrow.write(out_path, summary; compress=:zstd)

    @info "Done" rows=nrow(summary) path=out_path
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: aggregate_layer1.jl <out_arrow> @<shards_list_file>")
        println("       aggregate_layer1.jl <out_arrow> <shard1> [shard2 ...]")
        exit(1)
    end
    out_path = ARGS[1]
    rest     = ARGS[2:end]
    shard_paths = if length(rest) == 1 && startswith(rest[1], "@")
        readlines(rest[1][2:end])
    else
        rest
    end
    main(out_path, shard_paths)
end
