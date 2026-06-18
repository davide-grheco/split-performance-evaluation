using CairoMakie
using ColorSchemes
using DataFrames

export cd_diagram, nemenyi_heatmap, plot_per_dataset_spread
export DEFAULT_COLOR_SCHEME

const DEFAULT_COLOR_SCHEME = :glasbey_bw_minc_20_maxl_70_n256

"""
    cd_diagram(res, labels; title, height, width, color_scheme) -> Figure

Critical-difference diagram for a `NemenyiResult`. Algorithms are sorted by average rank;
horizontal bars connect methods not statistically distinguishable at the α level used
when computing `res`.
"""
function cd_diagram(res::NemenyiResult, labels::Vector{<:AbstractString};
    title::AbstractString="CD Diagram",
    height::Int=500, width::Int=2000,
    color_scheme::Symbol=DEFAULT_COLOR_SCHEME,
    legend_labelsize::Int=20)

    R = res.avg_ranks
    k = length(R)
    @assert length(labels) == k "labels length must match number of algorithms"

    perm = sortperm(R)
    r    = R[perm]
    labs = labels[perm]
    cd   = res.cd

    keep          = find_homogeneous_rank_spans(r, cd)
    placed, y_max = layout_group_bar_rows(keep)
    palette       = colorschemes[color_scheme]

    fig = Figure(size=(width, height), fontsize=36)
    ax  = Axis(fig[1, 1];
        title=title,
        xlabel="Average rank (lower = better)",
        yticksvisible=false,
        ygridvisible=false,
        ylabelvisible=false,
        yticklabelsvisible=false,
        yautolimitmargin=(0.15, 0.15),
    )

    lines!(ax, [1, k], [0.5, 0.5]; color=:black)

    x0 = 1.1
    x1 = min(k, x0 + cd)
    y0 = y_max + 0.25
    lines!(ax, [x0, x1], [y0, y0]; linewidth=5, color=:grey)
    text!(ax, "CD = $(round(cd, digits=3)) (α=0.05)";
        position=((x0 + x1) / 2, y0 + 0.10),
        align=(:center, :bottom), fontsize=20, color=:grey)

    for (x1s, x2s, y) in placed
        lines!(ax, [x1s, x2s], [y, y]; linewidth=5, color=:black)
    end

    for i in 1:k
        scatter!(ax, [r[i]], [0.5]; markersize=20, color=palette[i])
        lines!(ax, [r[i], r[i]], [0.35, 0.65]; color=palette[i])
    end

    Legend(fig[1, 2],
        [MarkerElement(color=palette[i], marker=:circle) for i in 1:k],
        labs; labelsize=legend_labelsize, patchsize=(18, 18), framevisible=false)

    return fig
end

"""
    nemenyi_heatmap(res, labels; alpha, show_values) -> Figure

Heatmap of pairwise Nemenyi p-values. Algorithms are ordered by average rank (best first).
Green (high p) = not distinguishable; red (low p) = significant difference.
"""
function nemenyi_heatmap(res::NemenyiResult, labels::AbstractVector{<:AbstractString};
    alpha::Real=0.05, show_values::Bool=true)

    k = length(labels)
    P = fill(1.0, k, k)
    for e in res.pairs
        P[e.i, e.j] = e.p_value
        P[e.j, e.i] = e.p_value
    end

    perm = sortperm(res.avg_ranks)
    labs = labels[perm]
    Pord = P[perm, perm]

    fig = Figure(size=(70k + 220, 70k + 160))
    ax  = Axis(fig[1, 1];
        title="Nemenyi pairwise p-values",
        xticks=(1:k, labs),
        yticks=(1:k, labs),
    )
    hm = heatmap!(ax, Pord; colormap=:coolwarm, colorrange=(0.0, 1.0))
    Colorbar(fig[1, 2], hm; label="p-value")
    if show_values
        for i in 1:k, j in 1:k
            text!(ax, string(round(Pord[i, j]; digits=3));
                position=(j, i), align=(:center, :center), color=:black, fontsize=10)
        end
    end
    return fig
end

"""
    plot_per_dataset_spread(df; metric_label, sortby, rev, fig_size, markersize, cap_h, line_w, color_scheme) -> Figure

Per-dataset spread chart: horizontal min–max ranges for benchmark performance (solid) and
estimation bias Δ (dashed), with per-splitter scatter overlaid and coloured by family.
"""
function plot_per_dataset_spread(df::DataFrame;
    metric_label::String="Metric",
    sortby::Symbol=:err_range, rev::Bool=true,
    fig_size=(1350, 750), markersize=7, cap_h=0.16, line_w=4,
    color_scheme::Symbol=DEFAULT_COLOR_SCHEME,
    family_colors::Union{Nothing,AbstractDict}=nothing,
    family_markers::Union{Nothing,AbstractDict}=nothing)

    df2      = collapse_dataset_algorithm(df)
    df2.err  = df2.diff

    dsum = per_dataset_spread_df(df2)
    sort!(dsum, sortby; rev=rev)

    datasets = String.(dsum.dataset)
    n        = nrow(dsum)
    y        = 1:n
    ymap     = Dict(datasets[i] => i for i in 1:n)
    df2.y    = [ymap[String(d)] for d in df2.dataset]

    df2.family = splitter_family.(String.(df2.algorithm))
    families   = sort(unique(df2.family))
    mcycle     = [:circle, :utriangle, :rect, :diamond, :cross, :xcross]
    palette    = colorschemes[color_scheme]
    markers    = family_markers === nothing ?
        Dict(f => mcycle[mod1(i, length(mcycle))] for (i, f) in enumerate(families)) :
        Dict(f => family_markers[f] for f in families)
    colors     = family_colors === nothing ?
        Dict(f => palette[i] for (i, f) in enumerate(families)) :
        Dict(f => family_colors[f] for f in families)

    fig = Figure(size=fig_size, fontsize=12)
    ax  = Axis(fig[1:2, 1];
        yticks=(y, datasets),
        xlabel="Value (solid = benchmark, dashed = bias Δ)",
        ylabel="Dataset",
        title="Per-dataset spread — $metric_label",
        subtitle="Ranges are min–max across splitters",
    )
    ax.ygridvisible = true

    _finite_xs = filter(isfinite, vcat(dsum.bench_min, dsum.err_min, dsum.bench_max, dsum.err_max))
    if isempty(_finite_xs)
        return fig
    end
    xmin = minimum(_finite_xs)
    xmax = maximum(_finite_xs)
    xpad = 0.04 * (xmax - xmin + eps())
    xlims!(ax, xmin - xpad, xmax + xpad)

    for i in 1:n
        yy       = y[i]
        x1b, x2b = dsum.bench_min[i], dsum.bench_max[i]
        lines!(ax, [x1b, x2b], [yy, yy]; linewidth=line_w, color=:black)
        lines!(ax, [x1b, x1b], [yy - cap_h, yy + cap_h]; linewidth=2, color=:black)
        lines!(ax, [x2b, x2b], [yy - cap_h, yy + cap_h]; linewidth=2, color=:black)
        x1e, x2e = dsum.err_min[i], dsum.err_max[i]
        lines!(ax, [x1e, x2e], [yy, yy]; linewidth=line_w, linestyle=:dash, color=:black)
        lines!(ax, [x1e, x1e], [yy - cap_h, yy + cap_h]; linewidth=2, color=:black)
        lines!(ax, [x2e, x2e], [yy - cap_h, yy + cap_h]; linewidth=2, color=:black)
    end

    for f in families
        sub = df2[df2.family .== f, :]
        scatter!(ax, sub.ips, sub.y .+ 0.2;
            marker=markers[f], color=colors[f], markersize=markersize,
            strokecolor=:black, strokewidth=0.6)
        scatter!(ax, sub.err, sub.y .+ 0.2;
            marker=markers[f], color=:transparent, strokecolor=colors[f],
            strokewidth=1.6, markersize=markersize)
    end

    Legend(fig[1, 2],
        [MarkerElement(marker=markers[f], color=colors[f], strokecolor=:black) for f in families],
        String.(families); title="Family", framevisible=false)
    Legend(fig[2, 2],
        [MarkerElement(marker=:line, color=:black, strokecolor=:black),
         MarkerElement(marker=:line, color=:transparent, strokecolor=:black)],
        ["Benchmark value", "Estimation bias Δ"]; framevisible=false)
    return fig
end
