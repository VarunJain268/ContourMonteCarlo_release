# Build the shared CMC initialisation, run either CMC method, and evaluate convergence.

using Distributions, LinearAlgebra, LaTeXStrings, Logging, Statistics, Plots, Measures, Printf

gr()
Plots.gr_legend_marker_to_line_factor[] = 1.5  # larger markers

include("utils.jl")
include("pdb_interface.jl")
include("targets.jl")
include("target_settings.jl")
include("chains.jl")
include("distances.jl")
include("convergence_metrics.jl")
include("umbrella.jl")
include("mbar.jl")
include("umbrella_mbar_pe_estimation.jl")
include("chmc.jl")
include("plotting.jl")
include("timing.jl")

cfgget(x::AbstractDict, name::Symbol, default) = haskey(x, name) ? x[name] : default
cfgget(x, name::Symbol, default) = hasproperty(x, name) ? getproperty(x, name) : default

function hmc_cfg(method_cfg, ctx)
    hmc = cfgget(method_cfg, :hmc, NamedTuple())
    eps = cfgget(hmc, :ϵ, ctx.ϵ_global)
    return (
        ϵ = eps,
        L = Int(cfgget(hmc, :L, ctx.L)),
    )
end

function configured_n_transitions(method_cfg, ctx)
    n = Int(cfgget(method_cfg, :n_samples, ctx.n_samples))
    n >= 0 || throw(ArgumentError("n_samples must be nonnegative, got $n"))
    return n
end

function prepend_initial_draws(ℓπ, out::NamedTuple, q_inits::AbstractMatrix{<:Real})
    chains = Vector{Any}(undef, length(out.chains))
    for i in eachindex(out.chains)
        c = out.chains[i]
        q0 = Vector{Float64}(view(q_inits, :, i))
        u0 = -LogDensityProblems.logdensity(ℓπ, q0)
        samples = Vector{Vector{Float64}}(undef, length(c.samples) + 1)
        samples[1] = q0
        for j in eachindex(c.samples)
            samples[j + 1] = Vector{Float64}(c.samples[j])
        end
        stats = vcat([(is_initial=true,)], c.stats)
        energies = (
            potential = vcat([Float64(u0)], Float64.(c.energies.potential)),
        )
        chains[i] = merge(c, (samples=samples, stats=stats, energies=energies))
    end
    n_kept = [length(c.samples) for c in chains]
    return merge(out, (chains=chains, n_kept=n_kept))
end

function matrix_chains_to_chain_outputs(chain_mats::AbstractVector, q0s::AbstractMatrix{<:Real})
    chains = Vector{Any}(undef, length(chain_mats))
    for i in eachindex(chain_mats)
        X = chain_mats[i]
        n = size(X, 2)
        samples = [Vector{Float64}(view(X, :, j)) for j in 1:n]
        chains[i] = (
            samples = samples,
            stats = [(is_initial=(j == 1),) for j in 1:n],
            θ0 = Vector{Float64}(view(q0s, :, i)),
            elapsed_sec = NaN,
        )
    end
    return (
        chains = chains,
        inits = [Vector{Float64}(view(q0s, :, i)) for i in axes(q0s, 2)],
        n_kept = [length(c.samples) for c in chains],
    )
end

function production_gradient_xaxis(chains)
    traces = [zeros(Int, length(c.samples)) for c in chains]
    for (i, c) in pairs(chains), t in 2:length(c.samples)
        s = c.stats[t]
        n_grads = Int(s.n_grad_evals)
        traces[i][t] = traces[i][t - 1] + n_grads
    end
    return aggregate_cumulative_int_traces(traces)
end

gradient_trace(gradient_xaxis, mode::Symbol) =
    mode === :avg_per_chain ? Float64.(gradient_xaxis.avg_per_chain) :
    mode === :total ? Float64.(gradient_xaxis.total) :
    error("gradient_axis_mode must be :avg_per_chain or :total, got :$mode")
gradient_grid(metrics, gradient_xaxis, mode::Symbol) = gradient_trace(gradient_xaxis, mode)[metrics.time_grid .+ 1]

function method_metrics(chains, ctx, report_name::Symbol)
    timed_metrics = @timed compute_metrics_over_time(chains, ctx.oracle_X, ctx.oracle_U;
        min_t=ctx.metrics_min_t, checkpoints=ctx.metrics_checkpoints, profile_timing=ctx.profile_timing,
        timing_label=String(report_name), projection_matrix=ctx.metric_projection_matrix,
        metrics_reductions=ctx.metrics_reductions,
        ℓπ=ctx.ℓπ, pe_hist_edges=ctx.pe_hist_edges,
        pe_kl_pseudocount=ctx.pe_kl_pseudocount)
    if ctx.profile_timing
        println(">>> Timing: $report_name metrics took $(round(timed_metrics.time, digits=3)) s")
    end
    metrics_by_reduction = timed_metrics.value
    metrics = getproperty(metrics_by_reduction, ctx.metrics_reduction)
    return metrics, metrics_by_reduction, timed_metrics.time
end

# Summarise warmup and production gradient costs for each chain.
function chain_gradient_costs(warmup, production)
    warm = Int.(warmup)
    prod = Int.(production)
    return (warmup_grad_evals_by_chain=warm, production_grad_evals_by_chain=prod,
        total_grad_evals_by_chain=warm .+ prod, warmup_grad_evals=sum(warm),
        production_grad_evals=sum(prod), total_grad_evals=sum(warm)+sum(prod))
end

function build_cmc_initialisation(ctx::NamedTuple)
    ctx.target_id === nothing && error("CMC methods require target_id or cmc_settings.target_id.")
    target_cfg = target_setting(ctx.target_id)
    umbrella_cfg = cfgget(ctx.cmc_settings, :umbrella, target_cfg.umbrella)
    pilot_cfg = cfgget(ctx.cmc_settings, :pilot_pool, NamedTuple())
    pilot_n_chains = Int(cfgget(pilot_cfg, :n_chains, 8))
    pilot_n_samples = Int(cfgget(pilot_cfg, :n_samples, 200))
    pilot_n_adapts = Int(cfgget(pilot_cfg, :n_adapts, 100))
    pilot_L = Int(cfgget(pilot_cfg, :L, ctx.L))
    pilot_delta = Float64(cfgget(pilot_cfg, :δ, 0.8))
    pilot_eps = cfgget(pilot_cfg, :ϵ, nothing)
    pilot_eps = pilot_eps === nothing ? nothing : Float64(pilot_eps)
    # ARMA unconstrained order: mu, phi, theta, log(sigma). Bounds affect starts only.
    pilot_init_bounds = ctx.target_id === :arma_11 ?
        (lower=[-2.0, -0.9, -0.9, -2.0], upper=[2.0, 0.9, 0.9, 2.0]) : nothing
    pilot_rng = ctx.method_rngs.cmc_initialisation
    pilot_rng_seed = ctx.rng_stream_seeds.cmc_initialisation

    before_pilot = read_target_grad_counter()
    pilot_pool, pilot_time = timed_exec("CMC pilot pool", ctx.profile_timing, () ->
        run_pilot_pool(pilot_rng, ctx.ℓπ;
            n_chains=pilot_n_chains,
            chain_workers=ctx.chain_workers,
            n_samples=pilot_n_samples,
            n_adapts=pilot_n_adapts,
            L=pilot_L,
            δ=pilot_delta,
            ϵ=pilot_eps, init_bounds=pilot_init_bounds)
    )

    pilot_grad_evals = read_target_grad_counter() - before_pilot

    mbar_seed = Int(cfgget(ctx.cmc_settings, :seed, experiment_stream_seed(ctx.seed, :cmc_mbar)))
    σu = Float64(umbrella_cfg.σu)

    mbar_out, mbar_time = timed_exec("CMC umbrella/MBAR", ctx.profile_timing, () ->
        run_mbar_general(
            seed=mbar_seed,
            cfg=target_cfg,
            lpi=ctx.ℓπ,
            u_min=umbrella_cfg.u_min,
            u_max=umbrella_cfg.u_max,
            σu=σu,
            K=Int(umbrella_cfg.K),
            replica_exchange=cfgget(ctx.cmc_settings, :replica_exchange, umbrella_cfg.replica_exchange),
            exchange_interval=cfgget(ctx.cmc_settings, :exchange_interval, umbrella_cfg.exchange_interval),
            pilot_pool=pilot_pool,
            nsteps=Int(cfgget(ctx.cmc_settings, :nsteps, umbrella_cfg.nsteps)),
            burn=Int(cfgget(ctx.cmc_settings, :burn, umbrella_cfg.burn)),
            eps_init_umbrella=Float64(something(ctx.ϵ_global, umbrella_cfg.eps_init_umbrella)),
            L_umbrella=Int(cfgget(ctx.cmc_settings, :L_umbrella, umbrella_cfg.L_umbrella)),
            adapt=Bool(cfgget(ctx.cmc_settings, :adapt, true)),
            target_acc=Float64(cfgget(ctx.cmc_settings, :target_acc, 0.8)),
            n_mbar_samples=ctx.n_chains,
            mbar_tol=Float64(cfgget(ctx.cmc_settings, :mbar_tol, 1e-8)),
            mbar_maxiters=Int(cfgget(ctx.cmc_settings, :mbar_maxiters, 1_000)),
            min_unique_fraction=cfgget(ctx.cmc_settings, :min_unique_fraction, nothing),
            display_plots=Bool(cfgget(ctx.cmc_settings, :display_mbar_plots, false)),
            defer_plots=Bool(cfgget(ctx, :defer_mbar_plots, false)),
            verbose=ctx.verbose,
        )
    )

    draws = mbar_out.mbar_draws
    q0s = draws.qs
    u0s = draws.us

    init = (
        q0s=q0s,
        u0s=u0s,
        mbar_plot_data=cfgget(ctx, :defer_mbar_plots, false) ? mbar_out.plot_data : nothing,
        mbar=(gradients=mbar_out.gradients, timing=mbar_out.timing,
            replica_exchange=mbar_out.replica_exchange, exchange_interval=mbar_out.exchange_interval,
            exchange_diagnostics=mbar_out.exchange_diagnostics),
        target_grad_evals=(pilot_pool=pilot_grad_evals, umbrellas=mbar_out.gradients.umbrellas_total,
            total=pilot_grad_evals + mbar_out.gradients.umbrellas_total),
        rng_stream_seeds=(pilot=pilot_rng_seed, mbar=mbar_seed),
        timing=(pilot_pool=pilot_time, mbar=mbar_time, total=pilot_time + mbar_time),
    )
    return init
end

# Independent seeds use independent targets, RNGs and task-local gradient counters.
function prepare_cmc_initialisations(seeds, cfg; n_chains::Int, L::Int,
        cmc_settings::NamedTuple, seed_workers::Int,
        profile_timing::Bool, verbose::Bool=true)
    seed_values = Int.(collect(seeds))
    allunique(seed_values) || throw(ArgumentError("Preparation seeds must be unique."))
    validate_chain_workers(seed_workers)
    isempty(seed_values) && return Dict{Int,Any}()
    # Build/load external models serially, before starting any sampling tasks.
    targets = [make_target(cfg.target_name; dim=cfg.dim, pdb_name=cfg.pdb_name,
        cfg.target_kwargs...) for _ in seed_values]
    for target in targets
        check_chain_target(target, min(seed_workers, length(seed_values)))
    end
    signature = (target_id=cfgget(cmc_settings, :target_id, cfg.id),
        target_name=cfg.target_name, dim=cfg.dim, pdb_name=cfg.pdb_name,
        target_kwargs=cfg.target_kwargs, n_chains=n_chains, L=L,
        ϵ_global=cfg.global_stepsize, cmc_settings=cmc_settings)
    prepared = Vector{Any}(undef, length(seed_values))
    run_chain_jobs(length(seed_values); chain_workers=seed_workers) do i
        seed = seed_values[i]
        ctx = (seed=seed, target_id=signature.target_id, ℓπ=targets[i],
            n_chains=n_chains, L=L, ϵ_global=cfg.global_stepsize, chain_workers=1, cmc_settings=cmc_settings,
            method_rngs=(cmc_initialisation=experiment_rng(seed, :cmc_initialisation),),
            rng_stream_seeds=(cmc_initialisation=experiment_stream_seed(seed, :cmc_initialisation),),
            profile_timing=profile_timing, verbose=verbose, defer_mbar_plots=true)
        init = build_cmc_initialisation(ctx)
        # MBAR resets its local counter; report the complete preparation cost to the scheduler.
        target_grad_counter()[] = init.target_grad_evals.total
        prepared[i] = (seed=seed, signature=signature, init=init)
        verbose && println("CMC preparation complete: seed $seed ($(init.target_grad_evals.total) gradient calls)")
    end
    return Dict(seed_values[i] => prepared[i] for i in eachindex(seed_values))
end

function checked_prepared_cmc(prepared, seed, signature, target_dim)
    prepared.seed == seed || throw(ArgumentError("Prepared CMC seed does not match this experiment."))
    isequal(prepared.signature, signature) ||
        throw(ArgumentError("Prepared CMC configuration does not match this experiment."))
    init = prepared.init
    size(init.q0s) == (target_dim, signature.n_chains) && length(init.u0s) == signature.n_chains ||
        throw(ArgumentError("Prepared CMC state dimensions do not match this experiment."))
    return init
end

function run_cmc_hmc_method(ctx::NamedTuple, cmc_init::NamedTuple)
    report_name = :cmc_hmc
    println("\n>>> Running Production Chains: $report_name")
    method_cfg = cfgget(ctx.method_settings, report_name, NamedTuple())
    hmc = hmc_cfg(method_cfg, ctx)
    n_transitions = configured_n_transitions(method_cfg, ctx)
    if hmc.ϵ === nothing
        error("Provide method_settings[:cmc_hmc].hmc.ϵ or ϵ_global.")
    end

    # Sample production chains from the prepared CMC states.
    timed_chains = @timed run_multiple_chains(ctx.method_rngs.cmc_hmc, ctx.ℓπ;
        chain_workers = ctx.chain_workers,
        init_qs = cmc_init.q0s,
        n_samples = n_transitions,
        ϵ_global = hmc.ϵ,
        adapt_per_chain = false,
        n_adapts = 0,
        drop_warmup = false,
        L = hmc.L,
    )
    out = prepend_initial_draws(ctx.ℓπ, timed_chains.value, cmc_init.q0s)
    out = merge(out, (cmc_init=cmc_init, n_transitions=n_transitions))

    if ctx.profile_timing
        println(
            ">>> Timing: $report_name chains (n_chains=$(ctx.n_chains), n_transitions=$n_transitions)",
            " took $(round(timed_chains.time, digits=3)) s",
        )
    end

    # Evaluate convergence and place metrics on the gradient-count axis.
    metrics, metrics_by_reduction, metrics_time = method_metrics(out.chains, ctx, report_name)
    grad_eval_grid = gradient_grid(metrics, production_gradient_xaxis(out.chains), ctx.gradient_axis_mode)
    # Return metrics and summaries only; the full chain history is no longer needed.
    return (
        n_chains = length(out.chains),
        cmc_initialisation_target_grad_evals = cmc_init.target_grad_evals.total,
        metrics = metrics,
        metrics_by_reduction = metrics_by_reduction,
        grad_eval_grid = grad_eval_grid,
        gradient_axis_mode = ctx.gradient_axis_mode,
        metrics_reduction = ctx.metrics_reduction,
        metrics_reductions = ctx.metrics_reductions,
        reference_evaluation = ctx.reference_evaluation,
        reference_floor = ctx.reference_floor,
        reference_floors_by_reduction = ctx.reference_floors_by_reduction,
        execution = (
            chain_workers = ctx.chain_workers,
            blas_threads = BLAS.get_num_threads(),
        ),
        sampling_diagnostics = chain_gradient_costs(
            out.warmup_grad_evals,
            [chain.production_grad_evals for chain in out.chains],
        ),
        rng_stream_seeds = (
            initialisation = cmc_init.rng_stream_seeds,
            sampling = ctx.rng_stream_seeds.cmc_hmc,
        ),
        timing = (
            initialisation = cmc_init.timing.total,
            production = timed_chains.time,
            metrics = metrics_time,
        ),
    )
end

function run_cmc_chmc_hmc_method(ctx::NamedTuple, cmc_init::NamedTuple)
    report_name = :cmc_chmc_hmc
    println("\n>>> Running Production Chains: $report_name")
    method_cfg = cfgget(ctx.method_settings, report_name, NamedTuple())
    chmc_cfg = cfgget(method_cfg, :chmc, NamedTuple())
    hmc = hmc_cfg(method_cfg, ctx)
    n_transitions = configured_n_transitions(method_cfg, ctx)
    alpha = Float64(cfgget(method_cfg, :alpha, 0.5))
    0.0 <= alpha <= 1.0 || throw(ArgumentError("alpha must be in [0, 1], got $alpha"))
    h_chmc_raw = cfgget(chmc_cfg, :h, hmc.ϵ)
    h_chmc_raw === nothing && error("Provide method_settings[:cmc_chmc_hmc].chmc.h or ϵ_global.")
    h_chmc = Float64(h_chmc_raw)
    h_hmc = hmc.ϵ === nothing ? h_chmc : Float64(hmc.ϵ)

    # Sample production chains from the prepared CMC states.
    timed_chains = @timed run_chmc(ctx.method_rngs.cmc_chmc_hmc;
        chain_workers = ctx.chain_workers,
        ℓπ = ctx.ℓπ,
        us = cmc_init.u0s,
        q0s = cmc_init.q0s,
        S = n_transitions,
        h = h_chmc,
        L = Int(cfgget(chmc_cfg, :L, ctx.L)),
        alpha = alpha,
        h_hmc = h_hmc,
        L_hmc = hmc.L,
        predictor_mode = cfgget(chmc_cfg, :predictor_mode, :improved),
        constraint_tol = Float64(cfgget(chmc_cfg, :constraint_tol, 1e-6)),
        reversibility_atol = Float64(cfgget(chmc_cfg, :reversibility_atol, 1e-8)),
        reversibility_rtol = Float64(cfgget(chmc_cfg, :reversibility_rtol, 1e-6)),
        root_maxiters = Int(cfgget(chmc_cfg, :root_maxiters, 250)),
        secant_init_width = Float64(cfgget(chmc_cfg, :secant_init_width, 1e-6)),
        root_residual_tol = Float64(cfgget(chmc_cfg, :root_residual_tol, 1e-12)),
        root_xtol = Float64(cfgget(chmc_cfg, :root_xtol, 1e-12)),
        root_rtol = Float64(cfgget(chmc_cfg, :root_rtol, 1e-10)),
    )

    chmc_out = timed_chains.value
    out = matrix_chains_to_chain_outputs(chmc_out.chains, cmc_init.q0s)
    out = merge(out, (cmc_init=cmc_init, chmc=chmc_out, n_transitions=n_transitions))

    if ctx.profile_timing
        println(
            ">>> Timing: $report_name chains (n_chains=$(ctx.n_chains), n_transitions=$n_transitions",
            ", n_chmc=$(chmc_out.n_chmc), n_hmc=$(chmc_out.n_hmc))",
            " took $(round(timed_chains.time, digits=3)) s",
        )
    end
    if ctx.verbose && chmc_out.n_chmc > 0
        println("    CHMC acceptance (summary): ", summary_stats(chmc_out.chmc_accept_rate))
        println("    RATTLE fail rate (summary): ", summary_stats(chmc_out.rattle_fail_rate))
        println("    Mean ΔH (summary): ", summary_stats(chmc_out.mean_ΔH))
        println("    Max |U(q)-u| (summary): ", summary_stats(chmc_out.max_constraint_err))
    end

    # Evaluate convergence and place metrics on the gradient-count axis.
    metrics, metrics_by_reduction, metrics_time = method_metrics(out.chains, ctx, report_name)
    gradient_counts = gradient_trace(chmc_out.gradient_eval_xaxes, ctx.gradient_axis_mode)
    grad_eval_grid = gradient_counts[metrics.time_grid .+ 1]
    transition_marker_x = chmc_out.n_hmc > 0 ? gradient_counts[chmc_out.transition_step + 1] : nothing
    # Return metrics and summaries only; the full chain history is no longer needed.
    return (
        n_chains = length(out.chains),
        cmc_initialisation_target_grad_evals = cmc_init.target_grad_evals.total,
        metrics = metrics,
        metrics_by_reduction = metrics_by_reduction,
        grad_eval_grid = grad_eval_grid,
        gradient_axis_mode = ctx.gradient_axis_mode,
        transition_marker_x = transition_marker_x,
        metrics_reduction = ctx.metrics_reduction,
        metrics_reductions = ctx.metrics_reductions,
        reference_evaluation = ctx.reference_evaluation,
        reference_floor = ctx.reference_floor,
        reference_floors_by_reduction = ctx.reference_floors_by_reduction,
        execution = (
            chain_workers = ctx.chain_workers,
            blas_threads = BLAS.get_num_threads(),
        ),
        sampling_diagnostics = merge(
            chain_gradient_costs(
                zeros(Int, ctx.n_chains),
                chmc_out.target_grad_evals_by_chain,
            ),
            (
                reverse_check_rejections = chmc_out.reverse_check_rejections,
                constraint_solves_by_chain = chmc_out.constraint_solves_by_chain,
                reversibility_atol = chmc_out.reversibility_atol,
                reversibility_rtol = chmc_out.reversibility_rtol,
            ),
        ),
        rng_stream_seeds = (
            initialisation = cmc_init.rng_stream_seeds,
            sampling = ctx.rng_stream_seeds.cmc_chmc_hmc,
        ),
        timing = (
            initialisation = cmc_init.timing.total,
            production = timed_chains.time,
            metrics = metrics_time,
        ),
    )
end

# Run the requested CMC methods
function run_cmc_experiment(
    seed::Int,

    # Target selection
    target_name::Symbol;
    target_id::Union{Nothing,Symbol},
    dim::Int,
    pdb_name::String,
    target_kwargs::NamedTuple,

    # Core experiment parameters
    n_chains::Int,
    chain_workers::Int,
    n_samples::Int,
    L::Int,
    methods::Vector{Symbol},

    # Oracle/reference parameters.
    oracle_size::Int,
    oracle_ref_path::Union{Nothing,String},

    # Adaptation/method-specific controls.
    ϵ_global::Union{Nothing,Real},
    method_settings,
    cmc_settings::NamedTuple,
    prepared_cmc,

    # Convergence-metric controls.
    metrics_min_t::Int  =  1,
    metrics_checkpoints::Vector{Int}  =  Int[],
    metrics_n_projections::Int,
    metrics_reductions,
    pe_nbins::Int,
    pe_kl_pseudocount::Float64  =  1e-12,
    gradient_axis_mode::Symbol,
    
    # Timing profiling
    profile_timing::Bool,
    verbose::Bool=true,
)

    # Set up independent random streams and validate the experiment.
    validate_chain_workers(chain_workers)
    setup_rng = MersenneTwister(seed)
    rng_metrics = MersenneTwister(seed + 1)
    method_stream_names = (:cmc_initialisation, :cmc_hmc, :cmc_chmc_hmc)
    rng_stream_seeds = NamedTuple{method_stream_names}(Tuple(
        experiment_stream_seed(seed, stream) for stream in method_stream_names
    ))
    method_rngs = NamedTuple{method_stream_names}(Tuple(
        MersenneTwister(getproperty(rng_stream_seeds, stream)) for stream in method_stream_names
    ))

    allowed_methods = Set([:cmc_hmc, :cmc_chmc_hmc])
    unknown_methods = [m for m in methods if !(m in allowed_methods)]
    if !isempty(unknown_methods)
        error("Unknown methods: $unknown_methods. Supported methods are: $(sort(collect(allowed_methods); by=String)).")
    end
    gradient_axis_mode in (:avg_per_chain, :total) ||
        error("gradient_axis_mode must be :avg_per_chain or :total, got :$gradient_axis_mode")
    metrics_reductions = normalise_metric_reductions(metrics_reductions)
    metrics_reduction = first(metrics_reductions)
    pe_nbins > 0 || error("pe_nbins must be positive, got $pe_nbins")
    pe_kl_pseudocount > 0 || error("pe_kl_pseudocount must be positive, got $pe_kl_pseudocount")
    cmc_target_id = cfgget(cmc_settings, :target_id, target_id)

    ℓπ = make_target(target_name; dim=dim, pdb_name=pdb_name, target_kwargs...)
    check_chain_target(ℓπ, chain_workers)
    target_dim = Int(LogDensityProblems.dimension(ℓπ))

    verbose && println("\n=======================================================")
    verbose && println("EXPERIMENT: Target=$target_name (D=$target_dim), Methods=$methods")
    verbose && println("Concurrent chain workers: $chain_workers (BLAS threads: $(BLAS.get_num_threads()))")
    verbose && println("=======================================================")
    if target_dim != dim
        println("Note: requested dim=$dim, but target resolved to D=$target_dim.")
    end

    # Build the shared reference samples and metric baselines.
    metric_projection_matrix = random_unit_vectors(rng_metrics, target_dim, metrics_n_projections)

    verbose && println(">>> Generating Oracle Samples...")
    oracle_cfg = (
        id = target_id,
        target_name = target_name,
        dim = target_dim,
        target_kwargs = target_kwargs,
        pdb_name = pdb_name,
        oracle_ref_path = oracle_ref_path,
    )
    oracle_X, _ = timed_exec("Oracle sampling", profile_timing, () ->
        load_oracle_samples(setup_rng, ℓπ, oracle_cfg; n_samples=oracle_size)
    )
    n_oracle = size(oracle_X, 2)
    verbose && println("Generated $n_oracle oracle samples.")
    oracle_U = potential_energies_matrix(ℓπ, oracle_X)

    n_eval = min(5000, n_oracle)
    metric_oracle_X = oracle_X[:, 1:n_eval]
    metric_oracle_U = oracle_U[1:n_eval]
    reference_evaluation = (
        n_eval = n_eval,
        n_base = n_oracle >= 10000 ? 5000 : 0,
        split = :first_5000_eval_next_5000_baseline,
        upper_quantile = 0.95,
    )
    pe_hist_edges = histogram_edges(metric_oracle_U; nbins=pe_nbins)
    verbose && println("Metric evaluation: first $n_eval reference draws; baseline pool: $(reference_evaluation.n_base) disjoint draws.")
    reference_floors_by_reduction = reference_floor_baselines(
        oracle_X, n_chains;
        rng = copy(rng_metrics),
        projection_matrix = metric_projection_matrix,
        metrics_reductions = metrics_reductions,
        oracle_Us = oracle_U,
        pe_hist_edges = pe_hist_edges,
        pe_kl_pseudocount = pe_kl_pseudocount,
    )
    reference_floor = getproperty(reference_floors_by_reduction, metrics_reduction)
    if verbose && isempty(propertynames(reference_floor))
        println("Reference floor skipped: need at least 10000 oracle samples and n_chains <= 5000.")
    end

    results = Dict{Symbol,Any}()
    method_ctx = (
        seed = seed,
        target_id = cmc_target_id,
        cmc_settings = cmc_settings,
        ℓπ = ℓπ,
        n_chains = n_chains,
        chain_workers = chain_workers,
        n_samples = n_samples,
        L = L,
        ϵ_global = ϵ_global,
        method_settings = method_settings,
        oracle_X = metric_oracle_X,
        oracle_U = metric_oracle_U,
        reference_evaluation = reference_evaluation,
        metrics_min_t = metrics_min_t,
        metrics_checkpoints = metrics_checkpoints,
        metrics_reduction = metrics_reduction,
        metrics_reductions = metrics_reductions,
        gradient_axis_mode = gradient_axis_mode,
        pe_hist_edges = pe_hist_edges,
        pe_kl_pseudocount = pe_kl_pseudocount,
        metric_projection_matrix = metric_projection_matrix,
        reference_floor = reference_floor,
        reference_floors_by_reduction = reference_floors_by_reduction,
        profile_timing = profile_timing,
        verbose = verbose,
        method_rngs = method_rngs,
        rng_stream_seeds = rng_stream_seeds,
    )

    cmc_init = if prepared_cmc === nothing
        nothing
    else
        signature = (
            target_id = cmc_target_id,
            target_name = target_name,
            dim = dim,
            pdb_name = pdb_name,
            target_kwargs = target_kwargs,
            n_chains = n_chains,
            L = L,
            ϵ_global = something(ϵ_global, target_setting(cmc_target_id).global_stepsize),
            cmc_settings = cmc_settings,
        )
        checked_prepared_cmc(prepared_cmc, seed, signature, target_dim)
    end

    # Run each method and attach its total runtime.
    if :cmc_hmc in methods
        if cmc_init === nothing
            cmc_init = build_cmc_initialisation(method_ctx)
        end
        cmc_hmc_res = run_cmc_hmc_method(method_ctx, cmc_init)
        GC.gc() # Reclaim chain histories before starting the next method.
        cmc_hmc_total = sum_times(cmc_hmc_res.timing.initialisation, cmc_hmc_res.timing.production)
        cmc_hmc_res = merge(cmc_hmc_res, (total_time=cmc_hmc_total,))
        results[:cmc_hmc] = cmc_hmc_res
        println("    Timing (cmc_hmc): initialisation=" * fmt_seconds(cmc_hmc_res.timing.initialisation) *
            ", production=" * fmt_seconds(cmc_hmc_res.timing.production) *
            ", total=" * fmt_seconds(cmc_hmc_total))
    end

    if :cmc_chmc_hmc in methods
        if cmc_init === nothing
            cmc_init = build_cmc_initialisation(method_ctx)
        end
        cmc_chmc_hmc_res = run_cmc_chmc_hmc_method(method_ctx, cmc_init)
        GC.gc()
        cmc_chmc_hmc_total = sum_times(cmc_chmc_hmc_res.timing.initialisation, cmc_chmc_hmc_res.timing.production)
        cmc_chmc_hmc_res = merge(cmc_chmc_hmc_res, (total_time=cmc_chmc_hmc_total,))
        results[:cmc_chmc_hmc] = cmc_chmc_hmc_res
        println("    Timing (cmc_chmc_hmc): initialisation=" * fmt_seconds(cmc_chmc_hmc_res.timing.initialisation) *
            ", production=" * fmt_seconds(cmc_chmc_hmc_res.timing.production) *
            ", total=" * fmt_seconds(cmc_chmc_hmc_total))
    end

    return results, ℓπ, oracle_X
end
