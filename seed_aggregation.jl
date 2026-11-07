# Combine metrics across seeds into median curves and uncertainty bands for plotting.

function mean_vectors(vs::Vector{<:AbstractVector{<:Real}})
    n = minimum(length(v) for v in vs)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        out[i] = mean_finite([v[i] for v in vs])
    end
    return out
end

function quantile_vectors(vs::Vector{<:AbstractVector{<:Real}}, q::Real)
    n = minimum(length(v) for v in vs)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        vals = finite_vec([v[i] for v in vs])
        out[i] = isempty(vals) ? NaN : quantile(vals, q)
    end
    return out
end

function common_time_grid(ms::Vector{ConvergenceMetrics})
    n = minimum(length(m.time_grid) for m in ms)
    tg = ms[1].time_grid[1:n]
    for m in ms[2:end]
        m.time_grid[1:n] == tg || error("Cannot aggregate convergence metrics with different time grids.")
    end
    return tg, n
end

function quantile_convergence_metrics(ms::Vector{ConvergenceMetrics}, q::Real)
    tg, n = common_time_grid(ms)
    vals = Any[tg]
    for f in fieldnames(ConvergenceMetrics)[2:end]
        push!(vals, quantile_vectors([getproperty(m, f)[1:n] for m in ms], q))
    end
    return ConvergenceMetrics(vals...)
end

median_convergence_metrics(ms::Vector{ConvergenceMetrics}) = quantile_convergence_metrics(ms, 0.5)

function result_reference_floor_for_reduction(result, reduction::Symbol)
    reduction = canonical_metric_reduction(reduction)
    return getproperty(result.reference_floors_by_reduction, reduction)
end

function aggregate_reference_floors(fs)
    isempty(fs) && return NamedTuple()
    names = unique(vcat([collect(propertynames(f)) for f in fs]...))
    out_names, out_vals = Symbol[], Any[]
    for name in names
        # Each replicate represents the same statistic as the plotted curve:
        # the median of one independently generated reference error per seed.
        all(f -> hasproperty(f, name) && hasproperty(getproperty(f, name), :values), fs) || continue
        simulations = [getproperty(f, name).values for f in fs]
        R = minimum(length(v) for v in simulations)
        R > 0 || continue
        vals = [median([v[i] for v in simulations]) for i in 1:R]
        push!(out_names, name)
        push!(out_vals, reference_error_summary(vals))
    end
    return NamedTuple{Tuple(out_names)}(Tuple(out_vals))
end

function method_n_chains(rs)
    result = first(rs)
    return hasproperty(result, :n_chains) ? result.n_chains : length(result.out.chains)
end

function method_shift_target_grad_evals(rs)
    vals = [Float64(hasproperty(result, :cmc_initialisation_target_grad_evals) ?
        result.cmc_initialisation_target_grad_evals : result.out.cmc_init.target_grad_evals.total)
        for result in rs]
    return (x_shift_target_grad_evals=(mean=mean(vals), n=length(vals), values=vals),)
end

function aggregate_over_seeds(seed_results::AbstractVector{<:AbstractDict})
    methods = unique(vcat([collect(keys(result)) for result in seed_results]...))
    aggregated_results = Dict{Symbol,Any}()

    for method in methods
        method_results = [result[method] for result in seed_results if haskey(result, method)]
        evaluation = method_results[1].reference_evaluation

        # Seeds must use the same reference evaluation and contain the requested reductions.
        all(result -> isequal(result.reference_evaluation, evaluation), method_results) ||
            error("Cannot aggregate method :$method with different reference evaluation settings.")
        reductions = result_metric_reductions(method_results[1])
        for result in method_results
            available_reductions = result_metric_reductions(result)
            all(reduction -> reduction in available_reductions, reductions) ||
                error("Cannot aggregate method :$method with inconsistent metric reductions.")
        end

        # Combine each reduction into a median curve and its interquartile band.
        metrics_per_reduction = map(reductions) do reduction
            [result_metrics_for_reduction(result, reduction) for result in method_results]
        end
        metric_lists = reduction_named_tuple(reductions, metrics_per_reduction)
        metrics_by_reduction = map(median_convergence_metrics, metric_lists)
        metrics_q25_by_reduction = map(metric_lists) do metrics
            quantile_convergence_metrics(metrics, 0.25)
        end
        metrics_q75_by_reduction = map(metric_lists) do metrics
            quantile_convergence_metrics(metrics, 0.75)
        end

        # Combine the simulated reference errors separately from the sampled curves.
        floor_summaries = map(reductions) do reduction
            floors = [
                result_reference_floor_for_reduction(result, reduction)
                for result in method_results
            ]
            aggregate_reference_floors(floors)
        end
        reference_floors_by_reduction = reduction_named_tuple(reductions, floor_summaries)

        primary_reduction = canonical_metric_reduction(method_results[1].metrics_reduction)
        reference_floor = getproperty(reference_floors_by_reduction, primary_reduction)
        gradient_grids = [result.grad_eval_grid for result in method_results]
        transition_markers = [
            result.transition_marker_x for result in method_results
            if hasproperty(result, :transition_marker_x) && result.transition_marker_x !== nothing
        ]

        combined = (
            metrics = getproperty(metrics_by_reduction, primary_reduction),
            metrics_q25 = getproperty(metrics_q25_by_reduction, primary_reduction),
            metrics_q75 = getproperty(metrics_q75_by_reduction, primary_reduction),
            metrics_by_reduction = metrics_by_reduction,
            metrics_q25_by_reduction = metrics_q25_by_reduction,
            metrics_q75_by_reduction = metrics_q75_by_reduction,
            n_seeds = length(method_results),
            reference_evaluation = evaluation,
            n_chains = method_n_chains(method_results),
            gradient_axis_mode = method_results[1].gradient_axis_mode,
            metrics_reduction = primary_reduction,
            metrics_reductions = reductions,
            grad_eval_grid = mean_vectors(gradient_grids),
        )

        # Add optional plot annotations only when they are available.
        if !isempty(transition_markers)
            combined = merge(combined, (transition_marker_x=mean(transition_markers),))
        end
        if !isempty(propertynames(reference_floor))
            combined = merge(combined, (
                reference_floor = reference_floor,
                reference_floors_by_reduction = reference_floors_by_reduction,
            ))
        end
        initialisation_cost = method_shift_target_grad_evals(method_results)
        aggregated_results[method] = merge(combined, initialisation_cost)
    end

    return aggregated_results
end
