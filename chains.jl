# Run HMC chains and the NUTS pilots used for tuning, with gradient counts for each chain.

if !@isdefined CachedPartialMomentumRefreshment
    include(joinpath(@__DIR__, "cached_momentum_refresh.jl"))
end

if !@isdefined run_chain_jobs
    include(joinpath(@__DIR__, "chain_execution.jl"))
end

using Random, LogDensityProblems, LinearAlgebra, AdvancedHMC, BSON, StatsFuns
import AdvancedHMC: UnitEuclideanMetric,
    Hamiltonian, find_good_stepsize, Leapfrog, HMCKernel, Trajectory, MultinomialTS,
    StepSizeAdaptor, NoAdaptation, sample,
    FixedNSteps, ClassicNoUTurn

function build_kernel(integrator; momentum_refresh::Real, L::Int, sampler::Symbol=:static, max_depth::Real=10)
    pm = CachedPartialMomentumRefreshment(Float64(momentum_refresh))
    termination = sampler === :static ? FixedNSteps(L) :
        sampler === :nuts ? ClassicNoUTurn(max_depth=max_depth, Δ_max=1000.0) :
        throw(ArgumentError("Internal kernel must be :static (CMC) or :nuts (pilot tuning)."))
    traj = Trajectory{MultinomialTS}(integrator, termination)
    return HMCKernel(pm, traj)
end

function build_adaptor(integrator; adapt_stepsize::Bool,
        n_adapts::Int, δ::Real, drop_warmup::Bool)
    if !adapt_stepsize || n_adapts <= 0
        return NoAdaptation(), 0, false
    end
    return StepSizeAdaptor(δ, integrator), n_adapts, drop_warmup
end

function extract_energies(stats)
    return (potential=[Float64(-s[:log_density]) for s in stats],)
end

# Sample from an explicit initial phase point z0 = (θ0, p0).
function sample_from_phasepoint(rng::AbstractRNG, hamiltonian::Hamiltonian, kernel::HMCKernel, z0, n_samples::Int,
    adaptor, n_adapts::Int; drop_warmup::Bool,
    initial_grad_evals::Int=0)

    @assert !(drop_warmup && (adaptor isa NoAdaptation)) "Cannot drop warmup samples if there is no adaptation phase."
    n_keep = n_samples - (drop_warmup ? n_adapts : 0)
    samples = Vector{typeof(z0.θ)}(undef, n_keep)
    stats = Vector{NamedTuple}(undef, n_keep)

    z = z0
    h = hamiltonian
    κ = kernel
    warmup_grad_evals = n_adapts > 0 ? initial_grad_evals : 0

    for i in 1:n_samples
        before_transition = read_target_grad_counter()
        t = AdvancedHMC.transition(rng, h, κ, z)
        z = t.z
        tstat = AdvancedHMC.stat(t)
        h, κ, isadapted = AdvancedHMC.Adaptation.adapt!(h, κ, adaptor, i, n_adapts, z.θ, tstat.acceptance_rate)
        transition_grad_evals = read_target_grad_counter() - before_transition
        tstat = merge(tstat, (is_adapt=isadapted, n_grad_evals=transition_grad_evals))

        if i <= n_adapts
            warmup_grad_evals += transition_grad_evals
        end

        if !drop_warmup || i > n_adapts
            j = i - (drop_warmup ? n_adapts : 0)
            if j == 1 && (!drop_warmup || n_adapts == 0)
                tstat = merge(tstat, (n_grad_evals=tstat.n_grad_evals + initial_grad_evals,))
            end
            samples[j] = z.θ
            stats[j] = tstat
        end
    end

    return (
        samples=samples,
        stats=stats,
        hamiltonian=h,
        kernel=κ,
        warmup_grad_evals=warmup_grad_evals,
    )
end

function checked_chain_initial(rng, ℓπ, q; random_init::Bool,
        init_box, max_attempts::Int)
    max_attempts > 0 || throw(ArgumentError("max_init_attempts must be positive."))
    for attempt in 1:max_attempts
        if all(isfinite,q)
            lp,g = LogDensityProblems.logdensity_and_gradient(ℓπ,q)
            isfinite(lp) && all(isfinite,g) && return q
        end
        random_init || throw(ArgumentError("Supplied init_q has a non-finite log density or gradient."))
        if attempt < max_attempts
            low,high = init_box
            q = low .+ (high-low).*rand(rng,length(q))
        end
    end
    error("No finite initial log density and gradient after $max_attempts attempts in init_box=$init_box.")
end

function run_single_chain(rng::AbstractRNG, ℓπ; n_samples::Int, init_q::Union{Nothing,AbstractVector},
    sampler::Symbol=:static, max_depth::Real=10, n_adapts::Int, L::Int,
    δ::Real, momentum_refresh::Real=0.0, adapt_stepsize::Bool,
    drop_warmup::Bool, ϵ::Union{Nothing,Real}, init_box::Tuple{<:Real,<:Real}=(-2.0, 2.0),
    validate_init::Bool=false, max_init_attempts::Int=1000)

    D = Int(LogDensityProblems.dimension(ℓπ))
    low, high = Float64(init_box[1]), Float64(init_box[2])
    high > low || throw(ArgumentError("init_box must satisfy high > low."))
    θ0 = init_q === nothing ? (low .+ (high - low) .* rand(rng, D)) : Vector(init_q)
    length(θ0) == D || throw(ArgumentError("init_q has length $(length(θ0)) but dim=$D"))

    setup_t0 = time_ns()
    before_setup = read_target_grad_counter()
    if validate_init
        θ0 = checked_chain_initial(rng, ℓπ, θ0; random_init=init_q === nothing,
            init_box=(low,high), max_attempts=max_init_attempts)
    end
    metric = UnitEuclideanMetric(D)
    hamiltonian = Hamiltonian(metric, ℓπ)
    ϵ0 = ϵ === nothing ? find_good_stepsize(rng, hamiltonian, θ0) : Float64(ϵ)
    setup_grad_evals = read_target_grad_counter() - before_setup
    setup_elapsed_sec = (time_ns() - setup_t0) / 1.0e9
    integrator = Leapfrog(ϵ0)
    kernel = build_kernel(integrator; momentum_refresh=momentum_refresh, L=L, sampler=sampler, max_depth=max_depth)
    adaptor, n_adapts_eff, drop_warmup_eff = build_adaptor(
        integrator;
        adapt_stepsize=adapt_stepsize,
        n_adapts=n_adapts,
        δ=δ,
        drop_warmup=drop_warmup,
    )

    timed = @timed begin
        before_initial = read_target_grad_counter()
        z0 = AdvancedHMC.phasepoint(rng, θ0, hamiltonian)
        initial_grad_evals = setup_grad_evals + read_target_grad_counter() - before_initial
        sample_from_phasepoint(
            rng, hamiltonian, kernel, z0, n_samples, adaptor, n_adapts_eff;
            drop_warmup=drop_warmup_eff,
            initial_grad_evals=initial_grad_evals,
        )
    end

    sampled = timed.value
    samples, stats = sampled.samples, sampled.stats
    energies = extract_energies(stats)
    step_size_used = stats[end][:nom_step_size]
    elapsed_sec = setup_elapsed_sec + timed.time
    production_stats = drop_warmup_eff ? stats : @view(stats[n_adapts_eff+1:end])
    production_grad_evals = sum((s.n_grad_evals for s in production_stats); init=0)
    total_grad_evals = sampled.warmup_grad_evals + production_grad_evals
    return (samples=samples, stats=stats, energies=energies, θ0=θ0, ϵ0=ϵ0,
        step_size_used=step_size_used,
        warmup_grad_evals=sampled.warmup_grad_evals, production_grad_evals=production_grad_evals,
        total_grad_evals=total_grad_evals, elapsed_sec=elapsed_sec)
end

function run_multiple_chains(rng::AbstractRNG, ℓπ; n_samples::Int, n_chains::Union{Nothing,Int}=nothing,
    init_qs::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    sampler::Symbol=:static, max_depth::Int=10, ϵ_global::Union{Nothing,Real}=nothing, L::Int=25, momentum_refresh::Real=0.0,
    adapt_per_chain::Bool, n_adapts::Int, δ::Real=0.8,
    drop_warmup::Bool, init_box::Tuple{<:Real,<:Real}=(-2.0, 2.0), chain_workers::Int,
    validate_init::Bool=false, max_init_attempts::Int=1000)

    D = Int(LogDensityProblems.dimension(ℓπ))
    if init_qs === nothing
        n_chains === nothing && throw(ArgumentError("Provide n_chains when init_qs is not given."))
        n_chains_final = Int(n_chains)
    else
        Dq, n_from_q = size(init_qs)
        Dq == D || throw(ArgumentError("Dimension mismatch in init_qs: got $Dq, expected $D."))
        n_chains_final = n_from_q
        if n_chains !== nothing && Int(n_chains) != n_chains_final
            throw(ArgumentError("n_chains=$(n_chains) does not match size(init_qs, 2)=$(n_chains_final)."))
        end
    end

    if !adapt_per_chain && ϵ_global === nothing
        throw(ArgumentError("ϵ_global must be provided when adapt_per_chain=false."))
    end

    validate_chain_workers(chain_workers)
    check_chain_target(ℓπ, chain_workers)
    chain_rngs = [MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains_final]
    chains = Vector{Any}(undef, n_chains_final)

    target_grad_evals_by_chain = run_chain_jobs(n_chains_final; chain_workers=chain_workers) do c
        q0 = init_qs === nothing ? nothing : Vector(@view init_qs[:, c])

        if adapt_per_chain
            chains[c] = run_single_chain(chain_rngs[c], ℓπ;
                n_samples=n_samples, init_q=q0, sampler=sampler, max_depth=max_depth, n_adapts=n_adapts, L=L,
                δ=δ, momentum_refresh=momentum_refresh, adapt_stepsize=true,
                drop_warmup=drop_warmup, ϵ=nothing, init_box=init_box,
                validate_init=validate_init, max_init_attempts=max_init_attempts)
        else
            chains[c] = run_single_chain(chain_rngs[c], ℓπ;
                n_samples=n_samples, init_q=q0, sampler=sampler, max_depth=max_depth, n_adapts=0, L=L,
                δ=δ, momentum_refresh=momentum_refresh, adapt_stepsize=false,
                drop_warmup=false, ϵ=Float64(ϵ_global), init_box=init_box,
                validate_init=validate_init, max_init_attempts=max_init_attempts)
        end
    end

    result = (
        chains = chains,
        chain_workers = chain_workers,
        target_grad_evals_by_chain = target_grad_evals_by_chain,
        adapt_per_chain = adapt_per_chain,
        ϵ_global = adapt_per_chain ? nothing : Float64(ϵ_global),
        step_sizes_reported = [o.step_size_used for o in chains],
        warmup_grad_evals = [o.warmup_grad_evals for o in chains],
        inits = [o.θ0 for o in chains],
        n_kept = [length(o.samples) for o in chains]
    )
    return result
end
