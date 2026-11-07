# Use NUTS pilots to tune the HMC step size and leapfrog count for each selected target.

using Random, Statistics, Printf

include(joinpath(@__DIR__, "pdb_interface.jl"))
include(joinpath(@__DIR__, "targets.jl"))
include(joinpath(@__DIR__, "target_settings.jl"))
include(joinpath(@__DIR__, "chains.jl"))
include(joinpath(@__DIR__, "utils.jl"))


# Tune a global step size using NUTS pilots and the median of final step sizes.
# Also returns the terminal adapted positions so the trajectory-length pilot can
# continue from the states reached during step-size adaptation.
function tune_global_stepsize_nuts(rng::AbstractRNG, ℓπ; n_pilots::Int, pilot_samples::Int, pilot_adapts::Int,
    δ::Real, max_depth::Int, chain_workers::Int)

    n_pilots > 0 || throw(ArgumentError("n_pilots must be positive."))
    pilot_samples > 0 || throw(ArgumentError("pilot_samples must be positive."))
    pilot_adapts >= 0 || throw(ArgumentError("pilot_adapts must be non-negative."))
    pilot_samples > pilot_adapts || throw(ArgumentError("pilot_samples must exceed pilot_adapts so at least one adapted draw is kept."))

    out = run_multiple_chains(rng, ℓπ;
        n_chains=n_pilots,
        chain_workers=chain_workers,
        n_samples=pilot_samples,
        sampler=:nuts,
        adapt_per_chain=true,
        n_adapts=pilot_adapts,
        δ=δ,
        max_depth=max_depth,
        drop_warmup=true,
        validate_init=true,
    )

    eps = Float64.(out.step_sizes_reported)
    adapted_qs = hcat([Vector{Float64}(ch.samples[end]) for ch in out.chains]...)
    return (eps_all = eps, eps_global = Float64(median(eps)), adapted_qs = adapted_qs)
end

# Tune the leapfrog count using fixed-step-size NUTS pilots from the adapted positions.
function tune_leapfrog_steps_nuts(rng::AbstractRNG, ℓπ; ϵ_global::Real, init_qs::AbstractMatrix{<:Real},
    pilot_samples::Int, δ::Real, max_depth::Int, chain_workers::Int)

    ϵ_global > 0 || throw(ArgumentError("ϵ_global must be positive."))
    pilot_samples > 0 || throw(ArgumentError("pilot_samples must be positive."))
    size(init_qs, 2) > 0 || throw(ArgumentError("init_qs must contain at least one pilot position."))

    taus = Float64[]
    n_steps_all = Int[]

    out = run_multiple_chains(rng, ℓπ;
        init_qs=init_qs,
        chain_workers=chain_workers,
        n_samples=pilot_samples,
        sampler=:nuts,
        adapt_per_chain=false,
        ϵ_global=Float64(ϵ_global),
        n_adapts=0,
        δ=δ,
        max_depth=max_depth,
        drop_warmup=false,
        validate_init=true,
    )

    n_steps_by_chain = Vector{Int}[]
    chain_summaries = NamedTuple[]
    for (chain, ch) in enumerate(out.chains)
        stats = ch.stats
        steps = Int[]

        for s in stats
            n_steps_i = Int(s.n_steps)
            n_steps_i > 0 || continue

            τ = Float64(ϵ_global) * n_steps_i
            isfinite(τ) || continue

            push!(taus, τ)
            push!(n_steps_all, n_steps_i)
            push!(steps, n_steps_i)
        end
        isempty(steps) && error("No valid NUTS step counts for chain $chain.")
        divergences = count(s -> s.numerical_error, stats)
        depth_hits = count(s -> s.tree_depth >= max_depth, stats)
        push!(n_steps_by_chain, steps)
        push!(chain_summaries, (chain=chain, transitions=length(stats),
            valid_steps=length(steps), p10=quantile(steps, 0.1), median=median(steps),
            p90=quantile(steps, 0.9), divergences=divergences,
            divergence_rate=divergences / length(stats), max_depth_hits=depth_hits,
            max_depth_hit_rate=depth_hits / length(stats)))
    end

    isempty(taus) && error("No valid NUTS stats collected for τ estimation.")

    τ_star = Float64(median(taus))
    L_star = max(1, round(Int, τ_star / Float64(ϵ_global)))

    return (tau_all = taus, tau_star = τ_star, L_star = L_star, n_steps_all = n_steps_all,
        n_steps_by_chain = n_steps_by_chain, chain_summaries = chain_summaries)
end

function print_tuning_summary(results; io::IO=stdout)
    isempty(results) && return nothing
    width = max(length("Target"), maximum(length(string(r.target_id)) for r in results))
    println(io, "\n================ Tuning summary ================")
    println(io, rpad("Target", width), "    D  Accept     Step size       L   Div (%)   Depth (%)")
    for r in results
        summaries = r.L_tune.chain_summaries
        n = sum(s.transitions for s in summaries)
        @printf(io, "%s  %3d   %.2f  %12.6g  %6d  %8.3f  %10.3f\n",
            rpad(string(r.target_id), width), r.dim, r.target_accept,
            r.nuts_tune.eps_global, r.L_tune.L_star,
            100 * sum(s.divergences for s in summaries) / n,
            100 * sum(s.max_depth_hits for s in summaries) / n)
    end
    println(io, "Div / Depth: percentages of trajectory-length pilot transitions.")
    return nothing
end

# Targets run sequentially; pilots within each stage may run concurrently.
# Add more IDs here, e.g. [:funnel_100d, :funnel_10d].
target_ids = [:funnel_10d]
chain_workers = Threads.nthreads(:default) # Set when starting Julia with --threads=N.
BLAS.set_num_threads(1) # Parallelise pilots rather than nested matrix operations.
n_pilots = 16
pilot_samples = 1001
pilot_adapts = 1000
length_pilot_samples = 1000
max_depth = 13
target_accept = 0.9
seed = 1

# Validate the whole list before starting any expensive pilots.
isempty(target_ids) && throw(ArgumentError("Specify at least one target in target_ids."))
tuning_configs = [target_setting(id) for id in unique(target_ids)]
tuning_results = NamedTuple[]
let
    for (target_id, cfg) in zip(unique(target_ids), tuning_configs)
        println("\nTarget: $(cfg.label) (id=$(target_id)), target_accept=$(target_accept), chain_workers=$(chain_workers)")
        ℓπ = build_target(cfg)
        rng = MersenneTwister(seed)
        println("Step-size tuning: $n_pilots pilots, $chain_workers workers")
        rng_nuts = MersenneTwister(rand(rng, UInt))
        nuts_tune = tune_global_stepsize_nuts(rng_nuts, ℓπ; n_pilots=n_pilots,
            pilot_samples=pilot_samples, pilot_adapts=pilot_adapts,
            δ=target_accept, max_depth=max_depth, chain_workers=chain_workers)
        println("NUTS tuning eps_all: $(nuts_tune.eps_all)")
        println("NUTS tuning eps_median: $(nuts_tune.eps_global)")

        println("Trajectory-length tuning: $(size(nuts_tune.adapted_qs, 2)) pilots, $chain_workers workers")
        rng_L = MersenneTwister(rand(rng, UInt))
        L_tune = tune_leapfrog_steps_nuts(rng_L, ℓπ; ϵ_global=nuts_tune.eps_global,
            init_qs=nuts_tune.adapted_qs, pilot_samples=length_pilot_samples,
            δ=target_accept, max_depth=max_depth, chain_workers=chain_workers)
        println("L tuning L_star: $(L_tune.L_star)")

        push!(tuning_results, (target_id=target_id, dim=Int(LogDensityProblems.dimension(ℓπ)),
            target_accept=target_accept, nuts_tune=nuts_tune, L_tune=L_tune))
    end
end
print_tuning_summary(tuning_results)
