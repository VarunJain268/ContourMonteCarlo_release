# Evaluate KS, CDF-ISE, and energy-KL at sampling checkpoints against reference draws.

struct ConvergenceMetrics
    time_grid::Vector{Int}
    dist_q_param_L2::Vector{Float64}
    dist_q_proj_L2::Vector{Float64}
    dist_q_param_KS::Vector{Float64}
    dist_q_proj_KS::Vector{Float64}
    dist_pe_KL::Vector{Float64}
end

# Allows us to compute time spent on different metrics if profile_timing=true
function time_block!(timings::Dict{Symbol,Float64}, profile_timing::Bool, key::Symbol, f::Function)
    if profile_timing
        t = @elapsed f()
        timings[key] = get(timings, key, 0.0) + t
    else
        f()
    end
    return nothing
end

function default_metric_checkpoints(n_transitions::Int)
    n_transitions >= 0 || error("n_transitions must be nonnegative.")
    return collect(0:n_transitions)
end

function reduce_metric_values(v::AbstractVector{<:Real}, reduction::Symbol)
    vals = finite_vec(v)
    isempty(vals) && return NaN
    reduction === :median && return median(vals)
    reduction === :maximum && return maximum(vals)
    throw(ArgumentError("metrics_reduction must be :median or :maximum, got :$reduction"))
end

canonical_metric_reduction(reduction::Symbol) =
    reduction === :median ? :median :
    reduction === :maximum ? :maximum :
    throw(ArgumentError("Metric reduction must be :median or :maximum, got :$reduction"))

function normalise_metric_reductions(metrics_reductions)
    requested = metrics_reductions isa Symbol ? (metrics_reductions,) : Tuple(metrics_reductions)
    isempty(requested) && throw(ArgumentError("metrics_reductions must not be empty."))
    return Tuple(unique(canonical_metric_reduction.(requested)))
end

function result_metric_reductions(result)
    return Tuple(propertynames(result.metrics_by_reduction))
end

function result_metrics_for_reduction(result, reduction::Symbol)
    reduction = canonical_metric_reduction(reduction)
    hasproperty(result.metrics_by_reduction, reduction) ||
        error("Result does not contain metrics for reduction :$reduction.")
    return getproperty(result.metrics_by_reduction, reduction)
end

reduction_named_tuple(reductions::Tuple, values) =
    NamedTuple{reductions}(Tuple(values))

function potential_energies_matrix(ℓπ, X::AbstractMatrix)
    us = Vector{Float64}(undef, size(X, 2))
    @inbounds for j in axes(X, 2)
        us[j] = -LogDensityProblems.logdensity(ℓπ, @view X[:, j])
    end
    return us
end

# Retain the simulated errors so reference thresholds can be recomputed/reaggregated.
function reference_error_summary(values)
    vals = Float64.(values)
    q50, q95 = quantile(vals, [0.5, 0.95])
    return (median=q50, q95=q95, values=vals)
end

function reference_floor_baselines(
    oracle_samples::Matrix{Float64},
    n_chains::Int;
    rng::AbstractRNG,
    n_eval::Int=5000,
    n_base::Int=5000,
    projection_matrix::AbstractMatrix,
    R::Int=500,
    metrics_reductions,
    oracle_Us::AbstractVector{<:Real},
    pe_hist_edges::AbstractVector{<:Real},
    pe_kl_pseudocount::Real,
)
    reductions = normalise_metric_reductions(metrics_reductions)
    empty_result = reduction_named_tuple(reductions, (NamedTuple() for _ in reductions))
    size(oracle_samples, 2) >= n_eval + n_base || return empty_result
    n_base >= n_chains || return empty_result

    dimension = size(oracle_samples, 1)
    reference_samples = @view oracle_samples[:, 1:n_eval]
    baseline_samples = @view oracle_samples[:, n_eval + 1:n_eval + n_base]
    projection_directions = Matrix{Float64}(projection_matrix)
    size(projection_directions, 1) == dimension || error(
        "projection_matrix has $(size(projection_directions, 1)) rows, expected D=$dimension",
    )
    n_projections = size(projection_directions, 2)

    parameter_reference = build_reference_cache(reference_samples)
    projection_reference = build_reference_cache(Matrix((reference_samples' * projection_directions)'))

    oracle_Us = Float64.(oracle_Us)
    reference_energies = @view oracle_Us[1:n_eval]
    baseline_energies = @view oracle_Us[n_eval + 1:n_eval + n_base]
    energy_bin_edges = Float64.(pe_hist_edges)
    reference_energy_probabilities = histogram_probabilities(
        reference_energies, energy_bin_edges; pseudocount=pe_kl_pseudocount,
    )

    projected_samples = Matrix{Float64}(undef, n_chains, n_projections)
    metric_fields = (
        :dist_q_param_L2,
        :dist_q_proj_L2,
        :dist_q_param_KS,
        :dist_q_proj_KS,
        :dist_pe_KL,
    )
    errors_by_reduction = Dict(
        reduction => Dict(field => Float64[] for field in metric_fields)
        for reduction in reductions
    )

    for _ in 1:R
        sample_indices = randperm(rng, n_base)[1:n_chains]
        parameter_metrics = per_param_metrics_against_cache(
            @view(baseline_samples[:, sample_indices]), parameter_reference,
        )
        mul!(projected_samples, (@view baseline_samples[:, sample_indices])', projection_directions)
        projection_metrics = per_column_metrics_against_cache(projected_samples, projection_reference)
        energy_kl = binned_kl_to_reference(
            @view(baseline_energies[sample_indices]), reference_energy_probabilities, energy_bin_edges;
            pseudocount=pe_kl_pseudocount,
        )

        for reduction in reductions
            errors = errors_by_reduction[reduction]
            push!(errors[:dist_q_param_L2], reduce_metric_values(parameter_metrics.l2, reduction))
            push!(errors[:dist_q_proj_L2], reduce_metric_values(projection_metrics.l2, reduction))
            push!(errors[:dist_q_param_KS], reduce_metric_values(parameter_metrics.ks, reduction))
            push!(errors[:dist_q_proj_KS], reduce_metric_values(projection_metrics.ks, reduction))
            push!(errors[:dist_pe_KL], energy_kl)
        end
    end

    summaries = map(reductions) do reduction
        errors = errors_by_reduction[reduction]
        values = Tuple(reference_error_summary(errors[field]) for field in metric_fields)
        NamedTuple{metric_fields}(values)
    end
    floors = reduction_named_tuple(reductions, summaries)
    return floors
end

function compute_metrics_over_time(
    chains,
    oracle_samples::Matrix{Float64},
    oracle_Us::AbstractVector{<:Real};
    checkpoints::Vector{Int},
    min_t::Int,
    profile_timing::Bool,
    timing_label::AbstractString,
    projection_matrix::AbstractMatrix,
    metrics_reductions,
    ℓπ,
    pe_hist_edges::AbstractVector{<:Real},
    pe_kl_pseudocount::Real,
)
    n_chains = length(chains)
    n_draws = length(chains[1].samples)
    dimension = length(chains[1].samples[1])
    n_transitions = n_draws - 1
    reductions = normalise_metric_reductions(metrics_reductions)

    transition_checkpoints = isempty(checkpoints) ?
        default_metric_checkpoints(n_transitions) :
        sort(unique(Int.(checkpoints)))

    min_t = max(0, min_t)
    filter!(transition_checkpoints) do transition_count
        0 <= transition_count <= n_transitions &&
            (transition_count == 0 || transition_count >= min_t)
    end
    isempty(transition_checkpoints) && (transition_checkpoints = [0])

    projection_directions = Matrix{Float64}(projection_matrix)
    size(projection_directions, 1) == dimension || error(
        "projection_matrix has $(size(projection_directions, 1)) rows, expected D=$dimension",
    )
    n_projections = size(projection_directions, 2)
    parameter_reference = build_reference_cache(oracle_samples)
    projection_reference = build_reference_cache(projection_directions' * oracle_samples)

    curve_fields = fieldnames(ConvergenceMetrics)[2:end]
    curves = Dict(
        reduction => Dict(field => Float64[] for field in curve_fields)
        for reduction in reductions
    )

    oracle_Us = Float64.(oracle_Us)
    energy_bin_edges = Float64.(pe_hist_edges)
    reference_energy_probabilities = histogram_probabilities(
        oracle_Us, energy_bin_edges; pseudocount=pe_kl_pseudocount,
    )

    positions = Matrix{Float64}(undef, n_chains, dimension)
    projected_positions = Matrix{Float64}(undef, n_chains, n_projections)
    potential_energies = Vector{Float64}(undef, n_chains)

    timings = Dict{Symbol,Float64}()

    for transition_count in transition_checkpoints
        sample_index = transition_count + 1

        time_block!(timings, profile_timing, :state_prepare, () -> begin
            @inbounds for chain_index in 1:n_chains
                position = chains[chain_index].samples[sample_index]
                for coordinate in 1:dimension
                    positions[chain_index, coordinate] = position[coordinate]
                end
                potential_energies[chain_index] = if hasproperty(chains[chain_index], :energies)
                    chains[chain_index].energies.potential[sample_index]
                else
                    -LogDensityProblems.logdensity(ℓπ, position)
                end
            end
            mul!(projected_positions, positions, projection_directions)
        end)

        time_block!(timings, profile_timing, :dist_state_proj, () -> begin
            projection_metrics = per_column_metrics_against_cache(
                projected_positions, projection_reference,
            )
            projection_cdf_ise, projection_ks = projection_metrics.l2, projection_metrics.ks
            for reduction in reductions
                curves_for_reduction = curves[reduction]
                push!(curves_for_reduction[:dist_q_proj_L2], reduce_metric_values(projection_cdf_ise, reduction))
                push!(curves_for_reduction[:dist_q_proj_KS], reduce_metric_values(projection_ks, reduction))
            end
        end)

        time_block!(timings, profile_timing, :dist_state_param, () -> begin
            parameter_metrics = per_column_metrics_against_cache(positions, parameter_reference)
            parameter_cdf_ise, parameter_ks = parameter_metrics.l2, parameter_metrics.ks
            for reduction in reductions
                curves_for_reduction = curves[reduction]
                push!(curves_for_reduction[:dist_q_param_L2], reduce_metric_values(parameter_cdf_ise, reduction))
                push!(curves_for_reduction[:dist_q_param_KS], reduce_metric_values(parameter_ks, reduction))
            end
        end)

        time_block!(timings, profile_timing, :dist_pe, () -> begin
            energy_kl = binned_kl_to_reference(
                potential_energies, reference_energy_probabilities, energy_bin_edges;
                pseudocount=pe_kl_pseudocount,
            )
            for reduction in reductions
                push!(curves[reduction][:dist_pe_KL], energy_kl)
            end
        end)
    end

    if profile_timing
        label = isempty(timing_label) ? "" : " (" * timing_label * ")"
        println(">>> Timing: metrics breakdown$label")
        for key in (:state_prepare, :dist_state_proj, :dist_state_param, :dist_pe)
            println("    $key: $(round(timings[key], digits=3)) s")
        end
        total = sum(values(timings))
        println("    TOTAL: $(round(total, digits=3)) s")
    end

    metric_values = map(reductions) do reduction
        values = (curves[reduction][field] for field in curve_fields)
        ConvergenceMetrics(transition_checkpoints, values...)
    end
    metric_sets = reduction_named_tuple(reductions, metric_values)
    return metric_sets
end
