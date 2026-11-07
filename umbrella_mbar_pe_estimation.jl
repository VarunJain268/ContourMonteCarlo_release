# Pilot sampling, umbrella/MBAR initialisation, and PE-fit plots for cmc.jl and main.jl.
using StatsBase: wsample

function run_pilot_pool(rng::AbstractRNG, ℓπ::LDP;
    n_chains::Int, n_samples::Int, n_adapts::Int, L::Int, δ::Float64,
    ϵ::Union{Nothing,Float64}, chain_workers::Int, init_bounds)

    adapt = ϵ === nothing
    all_qs = Vector{Vector{Float64}}()
    all_us = Float64[]
    validate_chain_workers(chain_workers)
    check_chain_target(ℓπ, chain_workers)
    bounds = if init_bounds === nothing
        nothing
    else
        lower, upper = Float64.(init_bounds.lower), Float64.(init_bounds.upper)
        d = Int(LogDensityProblems.dimension(ℓπ))
        length(lower) == length(upper) == d || throw(ArgumentError("Pilot bounds must match target dimension $d."))
        all(isfinite, lower) && all(isfinite, upper) && all(lower .< upper) ||
            throw(ArgumentError("Pilot bounds must be finite with lower < upper."))
        (lower=lower, upper=upper)
    end
    chain_rngs = [MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
    pilot_outputs = Vector{Any}(undef, n_chains)
    run_chain_jobs(n_chains; chain_workers=chain_workers) do c
        chain_rng = chain_rngs[c]
        q0 = bounds === nothing ? nothing :
            bounds.lower .+ (bounds.upper .- bounds.lower) .* rand(chain_rng, length(bounds.lower))
        out = run_single_chain(chain_rng, ℓπ; init_q=q0,
            n_samples=n_samples, n_adapts=(adapt ? n_adapts : 0), L=L, δ=δ,
            adapt_stepsize=adapt, drop_warmup=true, ϵ=ϵ)
        pilot_outputs[c] = out
    end
    for out in pilot_outputs
        for (i, q) in enumerate(out.samples)
            push!(all_qs, Vector{Float64}(q))
            push!(all_us, out.energies.potential[i])
        end
    end
    return (qs=all_qs, us=all_us)
end

# Pick the pilot position whose energy is closest to the window centre.
function pick_closest_q(state_pool, centre::Real)
    length(state_pool.qs) == length(state_pool.us) ||
        error("Pilot positions and energies must have matching lengths.")
    isempty(state_pool.us) && error("The pilot pool is empty.")
    _, index = findmin(abs.(Float64.(state_pool.us) .- Float64(centre)))
    return copy(state_pool.qs[index])
end

# Bin discrete MBAR weights for the PE-fit plot.
function weighted_histogram(u::Vector{Float64}, w::Vector{Float64}, edges::Vector{Float64})
    nb = length(edges) - 1
    counts = zeros(Float64, nb)
    Δ = edges[2] - edges[1]

    @inbounds for i in eachindex(u)
        ui = u[i]
        if ui < edges[1] || ui >= edges[end]
            continue
        end
        b = Int(floor((ui - edges[1]) / Δ)) + 1
        b = clamp(b, 1, nb)
        counts[b] += w[i]
    end

    centres = @. 0.5 * (edges[1:end-1] + edges[2:end])
    return (centres = centres, density = counts ./ Δ)
end

# Draw pooled indices as before, then copy only the selected positions from their windows.
function sample_mbar_draws(rng::AbstractRNG,
    mbar_out,
    qs_by_state::AbstractVector,
    energies_by_state::AbstractVector{<:AbstractVector{<:Real}},
    w_unbiased::Vector{Float64},
    n_samples::Int)

    n_samples >= 0 || throw(ArgumentError("n_samples must be >= 0"))
    length(qs_by_state) == length(energies_by_state) ||
        error("Positions and energies must have the same number of windows.")
    window_ends = cumsum(length.(energies_by_state))
    total_samples = sum(length, energies_by_state)
    total_samples > 0 || error("No retained umbrella samples available.")
    total_samples == length(mbar_out.u_all) ||
        error("Window sample counts must match the pooled MBAR energies.")

    dimension = nothing
    offset = 0
    for window in eachindex(energies_by_state)
        positions = qs_by_state[window]
        energies = energies_by_state[window]
        length(positions) == length(energies) ||
            error("Window $window has different numbers of positions and energies.")
        energies == @view(mbar_out.u_all[offset + 1:window_ends[window]]) ||
            error("Window energies are not aligned with the pooled MBAR energies.")
        for position in positions
            if dimension === nothing
                dimension = length(position)
            else
                length(position) == dimension || error("Retained positions must have the same dimension.")
            end
        end
        offset = window_ends[window]
    end

    indices = Vector{Int}(wsample(rng, 1:total_samples, w_unbiased, n_samples))
    selected_positions = Matrix{Float64}(undef, dimension, n_samples)
    for (column, index) in enumerate(indices)
        window = searchsortedfirst(window_ends, index)
        offset = window == 1 ? 0 : window_ends[window - 1]
        selected_positions[:, column] .= qs_by_state[window][index - offset]
    end
    return (
        us = Float64.(mbar_out.u_all[indices]),
        qs = selected_positions,
    )
end

# Draw oracle samples and compute their potential energies
function oracle_energies(rng::AbstractRNG, ℓπ::LDP, cfg::NamedTuple; n::Int)
    X = load_oracle_samples(rng, ℓπ, cfg; n_samples=n)
    return potential_energies_matrix(ℓπ, X)
end

compute_window_unique_fractions(energies_by_state::Vector{Vector{Float64}}) =
    [isempty(us) ? NaN : length(unique(us)) / length(us) for us in energies_by_state]

function filter_windows_by_unique_fraction(energies_by_state::Vector{Vector{Float64}},
    qs_by_state::AbstractVector,
    centres::AbstractVector{<:Real},
    min_unique_fraction::Union{Nothing,Float64})
    length(qs_by_state) == length(energies_by_state) ||
        error("qs_by_state and energies_by_state length mismatch: $(length(qs_by_state)) vs $(length(energies_by_state)).")
    if min_unique_fraction === nothing
        return (
            energies = energies_by_state,
            qs = qs_by_state,
            centres = centres,
        )
    end

    0.0 <= min_unique_fraction <= 1.0 ||
        throw(ArgumentError("min_unique_fraction must lie in [0, 1], got $min_unique_fraction"))

    unique_fracs = compute_window_unique_fractions(energies_by_state)
    keep = unique_fracs .>= min_unique_fraction
    any(keep) || error("Unique-fraction filter dropped all windows. Lower min_unique_fraction.")

    return (
        energies = energies_by_state[keep],
        qs = qs_by_state[keep],
        centres = centres[keep],
    )
end

function choose_anchor_window(pilot_pool, centres::AbstractVector{<:Real})
    isempty(pilot_pool.us) && error("pilot_pool is empty; cannot choose anchor window.")

    mid_idx = 0.5 * (length(centres) + 1)
    best_idx = first(eachindex(centres))
    best_gap = Inf
    best_mid_dist = Inf

    for k in eachindex(centres)
        gap = minimum(abs.(pilot_pool.us .- Float64(centres[k])))
        mid_dist = abs(k - mid_idx)
        if gap < best_gap || (isapprox(gap, best_gap; atol=1e-12, rtol=0.0) && mid_dist < best_mid_dist)
            best_idx = k
            best_gap = gap
            best_mid_dist = mid_dist
        end
    end

    return best_idx
end

function anchor_sweep_order(anchor_idx::Int, K::Int)
    order = Int[anchor_idx]
    for d in 1:max(anchor_idx - 1, K - anchor_idx)
        left = anchor_idx - d
        right = anchor_idx + d
        left >= 1 && push!(order, left)
        right <= K && push!(order, right)
    end
    return order
end

elapsed_seconds(t0_ns::UInt64) = (time_ns() - t0_ns) / 1.0e9

# Render on the calling task after concurrent numerical work has finished.
function plot_mbar_energy_fit(data; title="Energy marginal: $(data.label)")
    (; phat, oracle_Us) = data
    plt = plot(
        phat.centres, phat.density;
        lw=2,
        color=:blue,
        label="CMC energy estimate",
        legend=:topright,
        xlabel="u",
        ylabel="density",
        title=title,
    )
    if !isempty(oracle_Us)
        histogram!(plt, oracle_Us; bins=80, normalize=:pdf,
            color=:gray, alpha=0.35, label="Reference draws")
    end
    return plt
end

# Display already-computed fits on the main task; no additional sampling or MBAR work.
function plot_mbar_fit_row(plot_data, seeds; display_plot::Bool=true)
    length(plot_data) == length(seeds) || throw(ArgumentError("One MBAR fit is required per seed."))
    isempty(plot_data) && return nothing
    panels = [plot_mbar_energy_fit(data; title="Seed $seed") for (data, seed) in zip(plot_data, seeds)]
    for panel in panels[2:end]
        plot!(panel; legend=false)
    end
    row = plot(panels...; layout=(1, length(panels)), link=:all,
        size=(600 * length(panels), 450), left_margin=12Plots.mm, bottom_margin=12Plots.mm,
        plot_title="Energy marginal: $(first(plot_data).label)")
    display_plot && display(row)
    return row
end

# Run umbrella sampling + MBAR; plotting can be deferred without changing RNG consumption.
function run_mbar_general(; seed::Int,
    cfg::NamedTuple,
    lpi::LDP,
    u_min::Real, u_max::Real, σu::Real, K::Int,
    replica_exchange::Bool,
    exchange_interval::Int,
    pilot_pool,
    nsteps::Int,
    burn::Int,
    eps_init_umbrella::Real,
    L_umbrella::Int,
    adapt::Bool,
    target_acc::Float64,
    nbins::Int=50,  # Bins for the MBAR energy-fit plot.
    n_mbar_samples::Int,
    mbar_tol::Float64,
    mbar_maxiters::Int,
    min_unique_fraction::Union{Nothing,Float64},
    display_plots::Bool, defer_plots::Bool, verbose::Bool=true)

    pilot_pool === nothing && throw(ArgumentError("A pilot pool is required."))
    rng = MersenneTwister(seed)
    ℓπ = lpi
    exchange_interval > 0 || throw(ArgumentError("exchange_interval must be positive"))

    plot_needed = display_plots
    oracle_Us = plot_needed ? oracle_energies(rng, ℓπ, cfg; n=50_000) : Float64[]
    centres = collect(Base.range(Float64(u_min), Float64(u_max); length=K))

    energies_by_state = Vector{Vector{Float64}}(undef, K)
    qs_by_state = Vector{Vector{Vector{Float64}}}(undef, K)
    umbrella_window_times = zeros(Float64, K)

    reset_target_grad_counter!()
    umbrellas_t0_ns = time_ns()
    exchange_diagnostics = nothing
    if replica_exchange
        exchange_out = run_umbrella_replica_exchange(rng, ℓπ, centres, Float64(σu), pilot_pool;
            seed=seed, nsteps=nsteps, burn=burn, ϵ=Float64(eps_init_umbrella), L=L_umbrella,
            adapt=adapt, target_acc=target_acc, exchange_interval=exchange_interval, verbose=verbose)
        energies_by_state = exchange_out.energies
        qs_by_state = exchange_out.qs
        umbrella_window_times = exchange_out.times
        exchange_diagnostics = exchange_out.diagnostics
    else
        left_q_seed = nothing
        right_q_seed = nothing
        left_ϵ_seed = Float64(eps_init_umbrella)
        right_ϵ_seed = Float64(eps_init_umbrella)

        anchor_window_index = choose_anchor_window(pilot_pool, centres)
        window_run_order = anchor_sweep_order(anchor_window_index, K)
        anchor_q_start = pick_closest_q(pilot_pool, Float64(centres[anchor_window_index]))

        for k in window_run_order
            if k == anchor_window_index
                q_start = copy(anchor_q_start)
                ϵ_start = Float64(eps_init_umbrella)
            elseif k < anchor_window_index
                q_start = copy(left_q_seed)
                ϵ_start = left_ϵ_seed
            else
                q_start = copy(right_q_seed)
                ϵ_start = right_ϵ_seed
            end

            window_t0_ns = time_ns()
            outk = run_umbrella_chain_ahmc(
                rng, ℓπ, centres[k], σu;
                q_init=q_start,
                nsteps=nsteps,
                burn=burn,
                ϵ=ϵ_start,
                L=L_umbrella,
                adapt=adapt,
                target_acc=target_acc,
            )
            umbrella_window_times[k] = elapsed_seconds(window_t0_ns)

            isempty(outk.energies) && error("Window $k produced no retained energy samples.")
            energies_by_state[k] = outk.energies
            qs_by_state[k] = outk.qs
            if k == anchor_window_index
                left_q_seed = copy(outk.q_last)
                right_q_seed = copy(outk.q_last)
                left_ϵ_seed = outk.ϵ_final
                right_ϵ_seed = outk.ϵ_final
            elseif k < anchor_window_index
                left_q_seed = copy(outk.q_last)
                left_ϵ_seed = outk.ϵ_final
            else
                right_q_seed = copy(outk.q_last)
                right_ϵ_seed = outk.ϵ_final
            end
        end
    end
    wall_umbrellas_total = elapsed_seconds(umbrellas_t0_ns)

    filter_out = filter_windows_by_unique_fraction(energies_by_state, qs_by_state, centres, min_unique_fraction)
    energies_for_mbar = filter_out.energies
    qs_for_mbar = filter_out.qs
    centres_for_mbar = filter_out.centres

    mbar_solve_t0_ns = time_ns()
    mbar_out = mbar_free_energies(
        energies_for_mbar, centres_for_mbar, σu;
        tol=mbar_tol,
        maxiters=mbar_maxiters,
        m=10,
    )
    wall_mbar_solve = elapsed_seconds(mbar_solve_t0_ns)

    mbar_postprocess_t0_ns = time_ns()
    w_unbiased = mbar_unbiased_weights(mbar_out)
    plot_data = if plot_needed
        edges = collect(Base.range(u_min, u_max; length=nbins + 1))
        phat = weighted_histogram(mbar_out.u_all, w_unbiased, edges)
        (phat=phat, oracle_Us=oracle_Us, label=cfg.label)
    else
        nothing
    end
    mbar_draws = sample_mbar_draws(
        rng,
        mbar_out,
        qs_for_mbar,
        energies_for_mbar,
        w_unbiased,
        n_mbar_samples,
    )
    wall_mbar_postprocess = elapsed_seconds(mbar_postprocess_t0_ns)
    umbrella_grad_total = read_target_grad_counter()

    slowest_window_index = argmax(umbrella_window_times)
    slowest_window = (
        index = slowest_window_index,
        centre = Float64(centres[slowest_window_index]),
        time = umbrella_window_times[slowest_window_index],
    )
    timing = (
        wall = (
            total = wall_umbrellas_total + wall_mbar_solve + wall_mbar_postprocess,
            umbrellas_total = wall_umbrellas_total,
            umbrellas_by_window = copy(umbrella_window_times),
            umbrellas_longest = slowest_window.time,
            umbrellas_longest_window = slowest_window,
            mbar_solve = wall_mbar_solve,
            mbar_postprocess = wall_mbar_postprocess,
        ),
    )
    gradients = (umbrellas_total = umbrella_grad_total,)

    if display_plots && !defer_plots
        display(plot_mbar_energy_fit(plot_data))
    end

    return (
        mbar_draws = mbar_draws,
        plot_data = plot_data,
        replica_exchange = replica_exchange,
        exchange_interval = exchange_interval,
        exchange_diagnostics = exchange_diagnostics,
        timing = timing,
        gradients = gradients,
    )
end
