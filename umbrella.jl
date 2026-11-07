# Sample energy-biased umbrella windows, either independently or with replica exchange.

using Random
using LinearAlgebra
using Statistics
using LogDensityProblems
using AdvancedHMC
import AdvancedHMC: UnitEuclideanMetric, Hamiltonian, Leapfrog, StepSizeAdaptor, NoAdaptation, sample


@inline umbrella_bias(u::Float64, c::Float64, inv2σ2::Float64) = (u - c)^2 * inv2σ2

@inline function biased_potential_and_gradient(ℓπ::LDP, q::AbstractVector{<:Real}, c::Float64, σu::Float64)
    u, gU = U_and_∇U(ℓπ, q)
    if !(isfinite(u) && all(isfinite, gU))
        return Inf, fill!(similar(q), 0.0), u
    end
    α = 1.0 + (u - c) / (σu^2)
    V = u + 0.5 * ((u - c) / σu)^2
    return V, α .* gU, u
end

struct EnergyBiasedTarget{T}
    base::T
    centre::Float64
    σu::Float64
end

LogDensityProblems.dimension(ℓ::EnergyBiasedTarget) = LogDensityProblems.dimension(ℓ.base)
LogDensityProblems.capabilities(::Type{<:EnergyBiasedTarget}) = LogDensityProblems.LogDensityOrder{1}()
function LogDensityProblems.logdensity(ℓ::EnergyBiasedTarget, q::AbstractVector{<:Real})
    u = -LogDensityProblems.logdensity(ℓ.base, q)
    if !isfinite(u)
        return -1e300
    end
    V = u + 0.5 * ((u - ℓ.centre) / ℓ.σu)^2
    return isfinite(V) ? -V : -1e300
end

function LogDensityProblems.logdensity_and_gradient(ℓ::EnergyBiasedTarget, q::AbstractVector{<:Real})
    V, gV, _ = biased_potential_and_gradient(ℓ.base, q, ℓ.centre, ℓ.σu)
    if !(isfinite(V) && all(isfinite, gV))
        return -1e300, zeros(Float64, length(q))
    end
    return -V, -gV
end

function extract_unbiased_states_and_energies(ℓπ::LDP, samples, burn::Int)
    us = Float64[]
    qs = Vector{Vector{Float64}}()
    sizehint!(us, max(1, length(samples) - burn))
    sizehint!(qs, max(1, length(samples) - burn))
    for t in (burn + 1):length(samples)
        q = samples[t]
        u = -LogDensityProblems.logdensity(ℓπ, q)
        if isfinite(u)
            push!(us, u)
            push!(qs, copy(q))
        end
    end
    return (qs=qs, energies=us)
end

function run_umbrella_chain_ahmc(rng::AbstractRNG, ℓπ::LDP, c::Float64, σu::Float64;
    q_init::Vector{Float64},
    nsteps::Int,
    burn::Int,
    ϵ::Float64,
    L::Int,
    adapt::Bool,
    target_acc::Float64)

    nsteps > burn || throw(ArgumentError("Need nsteps > burn"))

    d = Int(LogDensityProblems.dimension(ℓπ))
    q = copy(q_init)
    biased_target = EnergyBiasedTarget(ℓπ, c, σu)

    metric = UnitEuclideanMetric(d)
    hamiltonian = Hamiltonian(metric, biased_target)
    integrator = Leapfrog(Float64(ϵ))
    kernel = build_kernel(integrator; momentum_refresh=0.0, L=L)
    adaptor = (adapt && burn > 0) ? StepSizeAdaptor(target_acc, integrator) : NoAdaptation()
    n_adapts = adapt ? burn : 0

    samples, stats = sample(
        rng,
        hamiltonian,
        kernel,
        q,
        nsteps,
        adaptor,
        n_adapts;
        drop_warmup=false,
        progress=false,
        verbose=false,
    )

    retained = extract_unbiased_states_and_energies(ℓπ, samples, burn)
    ϵ_final = Float64(stats[end][:nom_step_size]) 

    return (
        energies = retained.energies,
        qs = retained.qs,
        q_last = copy(samples[end]),
        ϵ_final = ϵ_final,
    )
end

# Warm up each umbrella window, then alternate HMC updates and neighbouring swaps.
# Store samples by window; track replica labels separately for swap diagnostics.
function run_umbrella_replica_exchange(rng, target, centres, sigma, pilot;
    seed::Int, nsteps::Int, burn::Int, ϵ::Float64, L::Int,
    adapt::Bool, target_acc::Float64, exchange_interval::Int, verbose::Bool=true)

    # Validate sampling and initialisation settings.
    0 <= burn < nsteps || throw(ArgumentError("Need 0 <= burn < nsteps"))
    exchange_interval > 0 || throw(ArgumentError("exchange_interval must be positive"))
    L > 0 && ϵ > 0 && isfinite(ϵ) ||
        throw(ArgumentError("L and step size must be positive"))
    sigma > 0 && isfinite(sigma) ||
        throw(ArgumentError("Umbrella width must be positive and finite"))
    0 < target_acc < 1 || throw(ArgumentError("target_acc must be between 0 and 1"))
    pilot === nothing && throw(ArgumentError("A pilot pool is required."))
    length(centres) >= 2 || throw(ArgumentError("Replica exchange requires at least two windows"))

    # Give each window its own RNG, Hamiltonian, and evolving state.
    n_windows = length(centres)
    anchor_window = choose_anchor_window(pilot, centres)
    window_order = anchor_sweep_order(anchor_window, n_windows)
    window_rngs = [MersenneTwister(rand(rng, UInt)) for _ in 1:n_windows]
    swap_rng = MersenneTwister(918273 + seed - 1)

    hamiltonians = Vector{Any}(undef, n_windows)
    kernels = Vector{Any}(undef, n_windows)
    phase_points = Vector{Any}(undef, n_windows)
    step_sizes = zeros(n_windows)
    window_times = zeros(n_windows)
    current_energies = zeros(n_windows)
    position_samples = [Vector{Float64}[] for _ in 1:n_windows]
    energy_samples = [Float64[] for _ in 1:n_windows]

    initial_position = pick_closest_q(pilot, centres[anchor_window])
    left_position = initial_position
    right_position = initial_position
    left_step_size = ϵ
    right_step_size = ϵ

    replica_labels = collect(1:n_windows)
    replica_visits = zeros(Int, n_windows, n_windows)
    swap_attempts = zeros(Int, n_windows - 1)
    swap_accepts = zeros(Int, n_windows - 1)

    # Warm up windows in order, passing starting states along each anchor branch.
    gradients_before_warmup = read_target_grad_counter()
    for window in window_order
        window_started = time_ns()
        if window == anchor_window
            position = copy(left_position)
            step_size = ϵ
        elseif window < anchor_window
            position = copy(left_position)
            step_size = left_step_size
        else
            position = copy(right_position)
            step_size = right_step_size
        end
        hamiltonian = Hamiltonian(UnitEuclideanMetric(length(position)),
            EnergyBiasedTarget(target, centres[window], sigma))
        integrator = Leapfrog(step_size)
        kernel = build_kernel(integrator; momentum_refresh=0.0, L=L)
        phase_point = AdvancedHMC.phasepoint(window_rngs[window], position, hamiltonian)
        adaptor = adapt && burn > 0 ? StepSizeAdaptor(target_acc, integrator) : NoAdaptation()

        if burn > 0
            warmup = sample_from_phasepoint(window_rngs[window], hamiltonian, kernel,
                phase_point, burn, adaptor, adapt ? burn : 0; drop_warmup=false)
            step_sizes[window] = AdvancedHMC.nom_step_size(warmup.kernel.τ.integrator)
            hamiltonians[window] = warmup.hamiltonian
            kernels[window] = warmup.kernel
            position = copy(warmup.samples[end])
            phase_points[window] = AdvancedHMC.phasepoint(
                window_rngs[window], position, hamiltonians[window])
            current_energies[window] = -LogDensityProblems.logdensity(target, position)
        else
            step_sizes[window] = step_size
            hamiltonians[window] = hamiltonian
            kernels[window] = kernel
            phase_points[window] = phase_point
            current_energies[window] = -LogDensityProblems.logdensity(target, position)
        end

        verbose && println("  exchange warmup W$window/$n_windows: centre=$(centres[window]), step size=$(step_sizes[window])")
        if window <= anchor_window
            left_position = copy(position)
            left_step_size = step_sizes[window]
        end
        if window >= anchor_window
            right_position = copy(position)
            right_step_size = step_sizes[window]
        end
        window_times[window] += (time_ns() - window_started) / 1e9
    end
    warmup_gradients = read_target_grad_counter() - gradients_before_warmup
    if verbose
        println("Umbrella warmup complete; gradients=$warmup_gradients")
        flush(stdout)
    end

    # Freeze adaptation and advance every window before attempting swaps.
    swap_gradients = 0
    production_steps = nsteps - burn
    n_rounds = cld(production_steps, exchange_interval)
    for round in 1:n_rounds
        for window in 1:n_windows
            window_started = time_ns()
            block_steps = min(exchange_interval, production_steps - (round - 1) * exchange_interval)
            for step in 1:block_steps
                transition = AdvancedHMC.transition(window_rngs[window],
                    hamiltonians[window], kernels[window], phase_points[window])
                phase_points[window] = transition.z
                energy = -LogDensityProblems.logdensity(target, phase_points[window].θ)
                current_energies[window] = energy

                # Retain each finite-energy sample before exchanging window states.
                if isfinite(energy)
                    push!(energy_samples[window], energy)
                    push!(position_samples[window], copy(phase_points[window].θ))
                end
            end
            window_times[window] += (time_ns() - window_started) / 1e9
        end

        # Alternate disjoint pairs: (1,2), (3,4), ... then (2,3), (4,5), ...
        let
            for left_window in (isodd(round) ? 1 : 2):2:n_windows-1
                right_window = left_window + 1
                swap_attempts[left_window] += 1
                isfinite(current_energies[left_window]) &&
                    isfinite(current_energies[right_window]) || continue

                bias(energy, centre) = 0.5 * ((energy - centre) / sigma)^2
                log_swap_ratio =
                    bias(current_energies[left_window], centres[left_window]) +
                    bias(current_energies[right_window], centres[right_window]) -
                    bias(current_energies[right_window], centres[left_window]) -
                    bias(current_energies[left_window], centres[right_window])
                # Independently check the algebra for the exchange ratio.
                @assert isapprox(log_swap_ratio,
                    (centres[right_window] - centres[left_window]) *
                    (current_energies[left_window] - current_energies[right_window]) / sigma^2;
                    atol=1e-10, rtol=1e-10)

                if log(rand(swap_rng)) < min(0.0, log_swap_ratio)
                    left_sample = copy(phase_points[left_window].θ)
                    right_sample = copy(phase_points[right_window].θ)
                    gradients_before_swap = read_target_grad_counter()
                    phase_points[left_window] = AdvancedHMC.phasepoint(
                        swap_rng, right_sample, hamiltonians[left_window])
                    phase_points[right_window] = AdvancedHMC.phasepoint(
                        swap_rng, left_sample, hamiltonians[right_window])
                    swap_gradients += read_target_grad_counter() - gradients_before_swap
                    current_energies[left_window], current_energies[right_window] =
                        current_energies[right_window], current_energies[left_window]
                    replica_labels[left_window], replica_labels[right_window] =
                        replica_labels[right_window], replica_labels[left_window]
                    swap_accepts[left_window] += 1
                end
            end
        end

        for window in 1:n_windows
            replica_visits[replica_labels[window], window] += 1
        end
        if verbose && (round % 1000 == 0 || round == n_rounds)
            println("Round $round/$n_rounds; gradients=$(read_target_grad_counter()); swaps=$(sum(swap_accepts))/$(sum(swap_attempts))")
            flush(stdout)
        end
    end

    diagnostics = (
        attempts=swap_attempts,
        accepts=swap_accepts,
        swap_gradients=swap_gradients,
        warmup_gradients=warmup_gradients,
        epsilons=step_sizes,
        replica_visits=replica_visits,
        exchange_interval=exchange_interval,
        rounds=n_rounds,
        production_steps=production_steps,
    )
    return (energies=energy_samples, qs=position_samples,
        times=window_times, diagnostics=diagnostics)
end
