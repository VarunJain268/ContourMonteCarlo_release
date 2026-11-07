# Plotting utilities for CMC experiments. 
const METHOD_ORDER = [:cmc_chmc_hmc, :cmc_hmc]
const COLOURS = [
    RGB(0xdc / 255, 0x39 / 255, 0x12 / 255), # CMC-HMC
    RGB(0xff / 255, 0x99 / 255, 0x00 / 255), # CMC-CHMC
]
const MARKERS = [:star5, :diamond]
const METHOD_STYLE_INDEX = Dict(:cmc_hmc => 1, :cmc_chmc_hmc => 2)

const PE_METRICS = [
    (1, :dist_pe_KL, L"\mathrm{KL}(\nu_{\mathcal{U}} \Vert \Pi_{\mathcal{U}})" * " (approx.)"),
]

metric_reduction_label(reduction::Symbol) =
    reduction in (:maximum, :max, :worst) ? "worst-case" : "median"

metric_ylabel(ylabel::AbstractString, reduction::Symbol) =
    replace(ylabel, ")" => ", $(metric_reduction_label(reduction)))")

function method_label(name::Symbol)
    name == :cmc_hmc       && return "CMC-HMC (ours)"
    name == :cmc_chmc_hmc  && return "CMC-CHMC (ours)"
    error("Unknown CMC method: $name")
end

cmc_gradient_axis_label(mode::Symbol) =
    mode === :avg_per_chain ? "Avg target gradient evaluations / chain" :
    mode === :total ? "Target gradient evaluations (total)" :
    "Target gradient evaluations"

function expand_methods(results)
    all(method -> method in METHOD_ORDER, keys(results)) ||
        error("Plots support only :cmc_hmc and :cmc_chmc_hmc.")
    return [method for method in METHOD_ORDER if haskey(results, method)]
end

function series_style(name::Symbol)
    i = METHOD_STYLE_INDEX[name]
    colour = COLOURS[i]
    marker = MARKERS[i]

    markerstrokewidth = 0
    markersize = 5

    return (
        colour=colour,
        marker=marker,
        lw=2.2,
        markersize=markersize,
        markercolor=colour,
        markerstrokecolor=colour,
        markerstrokewidth=markerstrokewidth,
        marker_stride=1,
    )
end

function metric_ribbon(y::AbstractVector{<:Real}, ylower, yupper)
    (ylower === nothing || yupper === nothing) && return nothing
    n = min(length(y), length(ylower), length(yupper))
    n == 0 && return nothing
    lower = Vector{Float64}(undef, n)
    upper = Vector{Float64}(undef, n)
    any_visible = false
    @inbounds for i in 1:n
        yi = Float64(y[i])
        lo = Float64(ylower[i])
        hi = Float64(yupper[i])
        if !(isfinite(yi) && isfinite(lo) && isfinite(hi) && lo > 0.0 && hi >= lo)
            lower[i] = NaN
            upper[i] = NaN
            continue
        end
        lower[i] = max(yi - lo, 0.0)
        upper[i] = max(hi - yi, 0.0)
        any_visible = true
    end
    return any_visible ? (lower, upper) : nothing
end

function plot_stride_indices(n::Int, line_stride::Int)
    n <= 0 && return Int[]
    stride = max(line_stride, 1)
    idx = collect(1:stride:n)
    idx[end] == n || push!(idx, n)
    return idx
end

function truncate_plot_data(x, y, ylower, yupper, max_grad_evals)
    max_grad_evals === nothing && return x, y, ylower, yupper
    cutoff = Float64(max_grad_evals)
    keep = x .<= cutoff
    return x[keep], y[keep],
        ylower === nothing ? nothing : ylower[keep],
        yupper === nothing ? nothing : yupper[keep]
end

function truncate_transition_x(transition_x, max_grad_evals)
    transition_x === nothing && return nothing
    max_grad_evals === nothing && return transition_x
    return Float64(transition_x) <= Float64(max_grad_evals) ? transition_x : nothing
end

function plot_metric!(plt, x, y, style; label::AbstractString, ylabel::AbstractString, xlabel::AbstractString, transition_x, ylower, yupper, line_stride::Int, show_legend::Bool)
    n = min(length(x), length(y), ylower === nothing ? typemax(Int) : length(ylower), yupper === nothing ? typemax(Int) : length(yupper))
    n == 0 && return nothing
    idx_line = plot_stride_indices(n, line_stride)
    x = x[idx_line]
    y = y[idx_line]
    ylower_plot = ylower === nothing ? nothing : ylower[idx_line]
    yupper_plot = yupper === nothing ? nothing : yupper[idx_line]
    ribbon = metric_ribbon(y, ylower_plot, yupper_plot)
    if ribbon !== nothing
        plot!(plt, x, y; label="", ribbon=ribbon, color=style.colour, linealpha=0.0,
            fillcolor=style.colour, fillalpha=0.14)
    end

    # Line 
    plot!(plt, x, y; label="", legend=(show_legend ? :topright : false), legend_columns=1, legendfontsize=13, color=style.colour, linestyle=:solid, lw=style.lw, xlabel=xlabel,
            xguidefontsize=18, ylabel=ylabel, yguidefontsize=18, grid=false)

    # Markers (no legend entry)
    idx = 1:style.marker_stride:length(x)
    plot!(plt, x[idx], y[idx]; label="", seriestype=:scatter, color=style.colour, marker=style.marker, markersize=style.markersize,
        markercolor=style.markercolor, markerstrokecolor=style.markerstrokecolor, markerstrokewidth=style.markerstrokewidth)

    # Legend-only entry
    plot!(plt, [NaN, NaN], [NaN, NaN]; label=label, seriestype=:path, color=style.colour, linestyle=:solid, lw=style.lw, marker=style.marker, 
        markersize=style.markersize + 5, markercolor=style.markercolor, markerstrokecolor=style.markerstrokecolor, markerstrokewidth=2)
    transition_x !== nothing && vline!(plt, [Float64(transition_x)]; color=style.colour, linestyle=:dash, linewidth=1.5, label="")

    return nothing
end

function plot_metric_reductions(results)
    reductions = nothing
    for result in values(results)
        available = result_metric_reductions(result)
        reductions = reductions === nothing ? collect(available) :
            [reduction for reduction in reductions if reduction in available]
    end
    isempty(something(reductions, Symbol[])) &&
        error("Results do not share a common metric reduction.")
    return Tuple(reductions)
end

function plot_metric_quantile_for_reduction(result, reduction::Symbol, quantile_name::Symbol)
    by_reduction_name = Symbol(quantile_name, :_by_reduction)
    if hasproperty(result, by_reduction_name)
        values = getproperty(result, by_reduction_name)
        return hasproperty(values, reduction) ? getproperty(values, reduction) : nothing
    end
    return nothing
end

function reference_floor(results, reduction::Union{Nothing,Symbol})
    for r in values(results)
        selected = reduction === nothing ?
            canonical_metric_reduction(r.metrics_reduction) :
            canonical_metric_reduction(reduction)
        floor = if hasproperty(r, :reference_floors_by_reduction) &&
                hasproperty(r.reference_floors_by_reduction, selected)
            getproperty(r.reference_floors_by_reduction, selected)
        else
            NamedTuple()
        end
        !isempty(propertynames(floor)) && return floor
    end
    return nothing
end

function plot_reference_floor!(plt, floor, field::Symbol)
    floor === nothing && return nothing
    hasproperty(floor, field) || return nothing
    b = getproperty(floor, field)
    mid = b.median
    upper = reference_floor_upper(floor, field)
    isfinite(mid) && mid > 0.0 && hline!(plt, [mid]; color=:black, linestyle=:dot, linewidth=2, label="Reference floor")
    isfinite(upper) && upper > 0.0 && hline!(plt, [upper]; color=:gray40, linestyle=:dot, linewidth=1.3, label="Reference 95th percentile")
    return nothing
end

function plot_x_shift(result; shift_right::Bool)
    shift_right || return 0.0
    hasproperty(result, :x_shift_target_grad_evals) || return 0.0
    shift = Float64(result.x_shift_target_grad_evals.mean)
    isfinite(shift) || return 0.0
    mode = result.gradient_axis_mode
    if mode === :avg_per_chain
        hasproperty(result, :n_chains) && result.n_chains !== missing && result.n_chains > 0 ||
            error("Cannot apply per-chain x shift without a positive n_chains value.")
        return shift / result.n_chains
    end
    return shift
end

function shifted_transition_x(result, shift::Real)
    hasproperty(result, :transition_marker_x) || return nothing
    result.transition_marker_x === nothing && return nothing
    return Float64(result.transition_marker_x) + Float64(shift)
end

function reference_floor_upper(floor, field::Symbol)
    floor === nothing && return NaN
    hasproperty(floor, field) || return NaN
    b = getproperty(floor, field)
    return Float64(b.q95)
end

function metric_hitting_time(result, field::Symbol, threshold::Real; shift_right::Bool,
        metrics_reduction::Union{Nothing,Symbol})
    metrics = metrics_reduction === nothing ?
        result.metrics : result_metrics_for_reduction(result, metrics_reduction)
    hasproperty(metrics, field) || return missing
    y = Float64.(getproperty(metrics, field))
    x_base = Float64.(result.grad_eval_grid)
    n = min(length(x_base), length(y))
    n == 0 && return missing
    shift = plot_x_shift(result; shift_right=shift_right)
    x = x_base[1:n] .+ shift
    thresh = Float64(threshold)
    isfinite(thresh) || return missing
    for i in 1:n
        yi = y[i]
        if isfinite(yi) && yi <= thresh
            return x[i]
        end
    end
    return missing
end

function fmt_hitting_time(x)
    x === missing && return "not reached"
    return @sprintf("%.4g", Float64(x))
end

function print_reference_band_hitting_times(results; shift_right::Bool,
        fields=(:dist_q_proj_KS, :dist_q_proj_L2),
        labels=("KS projection", "CDF-ISE projection"),
        metrics_reduction::Union{Nothing,Symbol}=nothing)
    available_reductions = plot_metric_reductions(results)
    selected_reduction = metrics_reduction === nothing ?
        first(available_reductions) : canonical_metric_reduction(metrics_reduction)
    selected_reduction in available_reductions ||
        error("Results do not contain metrics for reduction :$selected_reduction.")
    floor = reference_floor(results, selected_reduction)
    if floor === nothing
        println("Reference-band hitting times skipped: no reference floor found.")
        return nothing
    end
    thresholds = [reference_floor_upper(floor, f) for f in fields]
    println()
    println("Reference-band hitting times (reduction=:$selected_reduction; plot shift_right=$(shift_right); threshold=reference q95; raw excludes shift_right, shifted includes it)")
    println("Thresholds:")
    for (label, threshold) in zip(labels, thresholds)
        println("  $(label): ", isfinite(threshold) ? @sprintf("%.4g", threshold) : "missing")
    end
    println()
    header = rpad("method", 18) * rpad("shift_right", 14) *
        rpad(labels[1] * " raw", 22) *
        rpad(labels[2] * " raw", 28) *
        rpad("total raw", 18) *
        rpad(labels[1] * " shifted", 26) *
        rpad(labels[2] * " shifted", 32) *
        "total shifted"
    println(header)
    for method in expand_methods(results)
        r = results[method]
        hits_raw = [metric_hitting_time(r, f, threshold; shift_right=false,
            metrics_reduction=selected_reduction) for (f, threshold) in zip(fields, thresholds)]
        hits_shifted = [metric_hitting_time(r, f, threshold; shift_right=true,
            metrics_reduction=selected_reduction) for (f, threshold) in zip(fields, thresholds)]
        total_raw = any(ismissing, hits_raw) ? missing : maximum(Float64.(hits_raw))
        total_shifted = any(ismissing, hits_shifted) ? missing : maximum(Float64.(hits_shifted))
        shift = plot_x_shift(r; shift_right=true)
        println(rpad(String(method), 18) *
            rpad(@sprintf("%.4g", shift), 14) *
            rpad(fmt_hitting_time(hits_raw[1]), 22) *
            rpad(fmt_hitting_time(hits_raw[2]), 28) *
            rpad(fmt_hitting_time(total_raw), 18) *
            rpad(fmt_hitting_time(hits_shifted[1]), 26) *
            rpad(fmt_hitting_time(hits_shifted[2]), 32) *
            fmt_hitting_time(total_shifted))
    end
    return nothing
end

function cmc_x_shift(results; shift_right::Bool)
    shift_right || return nothing
    shifts = Float64[]
    for name in (:cmc_hmc, :cmc_chmc_hmc)
        haskey(results, name) || continue
        shift = plot_x_shift(results[name]; shift_right=true)
        isfinite(shift) && shift > 0.0 && push!(shifts, shift)
    end
    return isempty(shifts) ? nothing : sum(shifts) / length(shifts)
end

function plot_cmc_shift_marker!(plt, shift)
    shift === nothing && return nothing
    s = Float64(shift)
    isfinite(s) && s > 0.0 || return nothing
    lo, hi = Float64.(ylims(plt))
    isfinite(lo) && isfinite(hi) && lo > 0.0 && hi > lo || return nothing
    xlo, xhi = Float64.(xlims(plt))
    loglo, loghi = log10(lo), log10(hi)
    logspan = loghi - loglo
    arrow_log = loglo - 0.035 * logspan
    floor_log = arrow_log - 0.018 * logspan
    y_floor = 10.0 ^ floor_log
    ylims!(plt, (y_floor, hi))
    xlims!(plt, (min(xlo, 0.0), xhi))

    y = 10.0 ^ arrow_log
    y_label = 10.0 ^ (arrow_log + 0.035 * logspan)
    plot!(plt, [0.0, s], [y, y]; color=:gray45, linewidth=1.8, label="")
    scatter!(plt, [0.0], [y]; color=:gray45, marker=:ltriangle, markersize=5,
        markercolor=:gray45, markerstrokecolor=:gray45, label="")
    scatter!(plt, [s], [y]; color=:gray45, marker=:rtriangle, markersize=5,
        markercolor=:gray45, markerstrokecolor=:gray45, label="")
    annotate!(plt, s / 2, y_label, text("PE approx.", 14, :gray30))
    return nothing
end

function method_line_stride(method::Symbol, line_stride::Int, line_stride_cmc_chmc_hmc::Union{Nothing,Int})
    method === :cmc_chmc_hmc && line_stride_cmc_chmc_hmc !== nothing && return line_stride_cmc_chmc_hmc
    return line_stride
end

function plot_projection_metrics(results; line_stride::Int, line_stride_cmc_chmc_hmc::Union{Nothing,Int},
        shift_right::Bool, max_grad_evals, metrics_reduction::Symbol, show_legend::Bool=true, legend_strip::Bool=true)
    selected_reduction = canonical_metric_reduction(metrics_reduction)
    selected_reduction in plot_metric_reductions(results) ||
        error("Results do not contain metrics for reduction :$selected_reduction.")

    method_keys = expand_methods(results)
    floor = reference_floor(results, selected_reduction)
    cmc_shift = cmc_x_shift(results; shift_right=shift_right)

    # Combined KS + CDF-ISE projection plot with a shared legend strip.
    # Layout: KS top-left, CDF-ISE top-right, legend strip across the bottom.
    Plots.gr_legend_marker_to_line_factor[] = 1.5 
    projection_panel_specs = [
        (1, :dist_q_proj_KS, "KS (projection)"),
        (2, :dist_q_proj_L2, "CDF-ISE (projection)"),
    ]
    projections = plot(
        layout=legend_strip ? @layout([a b; c{0.19h}]) : (1, 2),
        size=(1700, 665),
        margin=10mm, left_margin=14mm, right_margin=10mm, top_margin=4mm,
        dpi=600, tickfontsize=16, yscale=:log10,
        foreground_color_legend=nothing, background_color_legend=nothing,
    )
    for (i, name) in enumerate(method_keys)
        m = result_metrics_for_reduction(results[name], selected_reduction)
        label = method_label(name)
        style = series_style(name)
        shift = plot_x_shift(results[name]; shift_right=shift_right)
        x = Float64.(results[name].grad_eval_grid) .+ shift
        xlabel = cmc_gradient_axis_label(results[name].gradient_axis_mode)
        transition_x = shifted_transition_x(results[name], shift)
        line_stride_method = method_line_stride(name, line_stride, line_stride_cmc_chmc_hmc)

        for (panel, field, ylabel_text) in projection_panel_specs
            y = getproperty(m, field)
            q25 = plot_metric_quantile_for_reduction(results[name], selected_reduction, :metrics_q25)
            q75 = plot_metric_quantile_for_reduction(results[name], selected_reduction, :metrics_q75)
            ylower = q25 === nothing ? nothing : getproperty(q25, field)
            yupper = q75 === nothing ? nothing : getproperty(q75, field)
            x_plot, y_plot, ylower_plot, yupper_plot = truncate_plot_data(x, y, ylower, yupper, max_grad_evals)
            plot_metric!(projections[panel], x_plot, y_plot, style; label=label, ylabel=metric_ylabel(ylabel_text, selected_reduction), xlabel=xlabel, transition_x=truncate_transition_x(transition_x, max_grad_evals), ylower=ylower_plot, yupper=yupper_plot, line_stride=line_stride_method, show_legend=show_legend)
        end
    end
    for (panel, field, _) in projection_panel_specs
        plot_reference_floor!(projections[panel], floor, field)
        plot_cmc_shift_marker!(projections[panel], cmc_shift)
    end

    if legend_strip
        function _add_method_entry!(plt, name)
            haskey(results, name) || return
            style = series_style(name)
            plot!(plt, [NaN, NaN], [NaN, NaN]; label=method_label(name), seriestype=:path,
                  color=style.colour, linestyle=:solid, lw=style.lw, marker=style.marker,
                  markersize=style.markersize + 5,
                  markercolor=style.markercolor, markerstrokecolor=style.markerstrokecolor,
                  markerstrokewidth=2)
        end
        _add_method_entry!(projections[3], :cmc_chmc_hmc)
        _add_method_entry!(projections[3], :cmc_hmc)
        plot!(projections[3], [NaN, NaN], [NaN, NaN]; label="Reference floor",
              seriestype=:path, color=:black, linestyle=:dot, linewidth=2)
        plot!(projections[3], [NaN, NaN], [NaN, NaN]; label="Reference 95th percentile",
              seriestype=:path, color=:gray40, linestyle=:dot, linewidth=1.3)

        plot!(projections[3]; legend=:top, legend_columns=2,
              legendfontsize=14,
              top_margin=2mm, bottom_margin=0mm,
              framestyle=:none, xticks=nothing, yticks=nothing, xlabel="", ylabel="", grid=false)
    end
    plot!(projections[1]; legend=false, right_margin=13mm, bottom_margin=10mm)
    plot!(projections[2]; legend=false, left_margin=13mm, bottom_margin=10mm)
    for panel in 1:2
        plot!(projections[panel]; xlims=(0.0, xlims(projections[panel])[2]))
    end
    return projections
end

function plot_energy_metrics(results, legend_plot; line_stride::Int, line_stride_cmc_chmc_hmc::Union{Nothing,Int},
        shift_right::Bool, max_grad_evals, metrics_reduction::Symbol, show_legend::Bool=true)
    selected_reduction = canonical_metric_reduction(metrics_reduction)
    method_keys = expand_methods(results)
    floor = reference_floor(results, selected_reduction)
    cmc_shift = cmc_x_shift(results; shift_right=shift_right)

    Plots.gr_legend_marker_to_line_factor[] = 3.0  # restore larger legend markers for the PE plot
    pe = plot(layout=(1,1), size=(800,520), margin=10mm, left_margin=8mm, right_margin=9mm, top_margin=4mm, dpi=600, tickfontsize=16, yscale=:log10,
              foreground_color_legend=nothing, background_color_legend=nothing)
    for (i, name) in enumerate(method_keys)
        m = result_metrics_for_reduction(results[name], selected_reduction)
        label = method_label(name)
        style = series_style(name)
        shift = plot_x_shift(results[name]; shift_right=shift_right)
        x = Float64.(results[name].grad_eval_grid) .+ shift
        xlabel = cmc_gradient_axis_label(results[name].gradient_axis_mode)
        transition_x = shifted_transition_x(results[name], shift)
        line_stride_method = method_line_stride(name, line_stride, line_stride_cmc_chmc_hmc)

        for (panel, field, ylabel) in PE_METRICS
            y = getproperty(m, field)
            q25 = plot_metric_quantile_for_reduction(results[name], selected_reduction, :metrics_q25)
            q75 = plot_metric_quantile_for_reduction(results[name], selected_reduction, :metrics_q75)
            ylower = q25 === nothing ? nothing : getproperty(q25, field)
            yupper = q75 === nothing ? nothing : getproperty(q75, field)
            x_plot, y_plot, ylower_plot, yupper_plot = truncate_plot_data(x, y, ylower, yupper, max_grad_evals)
            plot_metric!(pe[panel], x_plot, y_plot, style; label=label, ylabel=ylabel, xlabel=xlabel, transition_x=truncate_transition_x(transition_x, max_grad_evals), ylower=ylower_plot, yupper=yupper_plot, line_stride=line_stride_method, show_legend=show_legend)
        end
    end
    for (panel, field, _) in PE_METRICS
        plot_reference_floor!(pe[panel], floor, field)
        plot_cmc_shift_marker!(pe[panel], cmc_shift)
    end
    for panel in eachindex(pe.subplots)
        plot!(pe[panel]; xlims=(0.0, xlims(pe[panel])[2]))
    end
 
    for series in pe[1].series_list
        series[:label] = ""
    end
    if show_legend
        for series in legend_plot.series_list
            plot!(pe[1], [NaN, NaN], [NaN, NaN];
                  label=series[:label], color=series[:seriescolor],
                  linestyle=series[:linestyle], linewidth=series[:linewidth],
                  marker=series[:markershape], markersize=series[:markersize],
                  markercolor=series[:markercolor],
                  markerstrokecolor=series[:markerstrokecolor],
                  markerstrokewidth=series[:markerstrokewidth])
        end
        lo, hi = ylims(pe[1])
        if isfinite(lo) && isfinite(hi) && 0 < lo < hi
            loglo, loghi = log10(lo), log10(hi)
            plot!(pe[1]; ylims=(lo, 10.0^(loglo + (loghi - loglo) / 0.95)))
        end
    end
    plot!(pe; legend=show_legend ? :topright : false, legend_columns=2,
          legendfontsize=11, size=(950, 570), top_margin=4mm)
    if show_legend && pe[1] isa Plots.Subplot{Plots.GRBackend}
        Plots.prepare_output(pe)
        w, h = Plots.get_size(pe)
        canvas = Plots.GRViewport(0.0, min(1.0, w / h), 0.0, min(1.0, h / w))
        vp = Plots.gr_viewport_from_bbox(pe[1], Plots.plotarea(pe[1]), w * Plots.px, h * Plots.px, canvas)
        leg = Plots.gr_get_legend_geometry(vp, pe[1])
        x, y = Plots.gr_legend_pos(pe[1], leg, vp)
        xpos = (x - vp.xmin + leg.xoffset) / (vp.xmax - vp.xmin) - 0.005
        ypos = (y - vp.ymin) / (vp.ymax - vp.ymin)
        plot!(pe; legend=(xpos, ypos))
    end

    return pe
end


# Give each final figure its own native window instead of replacing the previous one.
const CMC_PLOT_WINDOW_ID = Ref(100)

function display_plot_window(figure)
    backend = Plots.GR
    backend.inqdspsize() # Initialise GR's default workstation before opening another.
    window_id = (CMC_PLOT_WINDOW_ID[] += 1)
    backend.openws(window_id, "", 411) # GR's Qt window viewer.
    backend.deactivatews(1)
    backend.activatews(window_id)
    try
        gui(figure)
    finally
        backend.deactivatews(window_id)
        backend.closews(1) # Remove the extra default window GR opens while rendering.
        backend.openws(1, "", 0)
        backend.activatews(1)
    end
    return nothing
end

# Two figures for main.jl: median/worst projection metrics and energy KL.
function plot_results_summary(results; display_plots::Bool=true, kwargs...)
    median = plot_projection_metrics(results; metrics_reduction=:median, kwargs...)
    worst = plot_projection_metrics(results; metrics_reduction=:maximum,
        merge((; kwargs...), (legend_strip=false,))...)
    pe = plot_energy_metrics(results, median[3]; metrics_reduction=:median, kwargs...)
    for series in median[3].series_list
        series[:label] == " " && (series[:label] = "")
    end
    projections = plot(plot(median[1]), plot(median[2]), plot(worst[1]), plot(worst[2]), plot(median[3]);
        layout=@layout([a b; c d; e{0.17h}]), size=(1700, 1150), dpi=600)
    for (i, title) in enumerate(("Median — KS", "Median — CDF-ISE",
                                 "Worst case — KS", "Worst case — CDF-ISE"))
        plot!(projections[i]; title=title, titlefontsize=16, top_margin=6mm,
              bottom_margin=8mm, legend=false)
    end
    if display_plots
        display(projections)
        display(pe)
    end
    return (projections=projections, pe=pe)
end
