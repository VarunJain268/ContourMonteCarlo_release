# Choose your settings below, then run this file to sample and plot CMC results.

if !@isdefined CMC_RUNTIME_LOADED
    include("cmc.jl")
    include("seed_aggregation.jl")
    const CMC_RUNTIME_LOADED = true
end

# Settings: EDIT THIS BLOCK
# Target-specific step sizes and umbrella settings come from target_settings.jl.
target_id = :funnel_100d                      # Use an ID from available_target_settings().
cfg = target_setting(target_id)
seeds = collect(1:5)                # Use [1] for one run, or several seeds to combine results.
methods = [:cmc_hmc, :cmc_chmc_hmc]          # Keep either method or both.
chain_workers = Threads.nthreads(:default)  # Concurrent production chains within each seed.
seed_workers = 2 #min(length(seeds), chain_workers) # Concurrent seed preparations (pilots, umbrellas, MBAR).
# seed_workers = 1 lowers peak memory; all seeds run, with production chains still parallel.
blas_threads = 1                            # Keep matrix operations serial when chains run in parallel.
verbose = false                            # Print detailed progress and diagnostics; false keeps main messages and overall timings.
profile_timing = false                        # Print the time spent on sampling and metrics.

# Sampling: each method starts from the same CMC initialisation for a given seed.
n_chains = 1000                             # Number of chains per method and seed.
n_samples_hmc = 1000                         # HMC transitions per chain, excluding the initial draw.
n_samples_chmc = 250                         # Total CHMC and optional HMC transitions per chain.
method_settings = Dict(
    :cmc_hmc => (
        hmc=(ϵ=cfg.global_stepsize, L=cfg.L), # HMC step size and leapfrog steps per transition.
        n_samples=n_samples_hmc,
    ),
    :cmc_chmc_hmc => (
        alpha=1.0,                          # Fraction of transitions using CHMC; any remainder uses HMC.
        chmc=(
            h=cfg.cmc_tuning.h_chmc,        # CHMC step size from the target preset.
            L=cfg.cmc_tuning.L_chmc,        # Constrained steps per CHMC transition.
            predictor_mode=:cheap,         # Starting estimate for the secant constraint solver.
        ),
        hmc=(ϵ=cfg.global_stepsize, L=cfg.L), # Settings for the optional HMC continuation.
        n_samples=n_samples_chmc,
    ),
)

# Initialisation: pilots provide starting positions for the umbrella windows.
cmc_settings = (
    target_id=target_id,
    umbrella=cfg.umbrella,                  # Window range, width, warmup and sampling settings.
    pilot_pool=(
        n_chains=24,                       # Number of pilot chains.
        n_samples=200,                     # Transitions per pilot, including adaptation.
        n_adapts=100,                      # Initial pilot transitions used for step-size adaptation.
        L=cfg.L,                           # Leapfrog steps per pilot transition.
    ),
    display_mbar_plots=true,                # Show the estimated energy distributions together.
    min_unique_fraction=0.6,               # Discard windows below this fraction of distinct energies.
)

# Metrics: compare each sampling checkpoint with reference draws.
oracle_size = 10_000                        # Requested reference draws; saved files supply their stored draws.
metrics_n_projections = 200                 # Random directions for projected KS and CDF-ISE.
metrics_reductions = (:median, :maximum)    # Keep both for the median and worst-case plot panels.
pe_nbins = 100                               # Histogram bins for the energy-KL estimate.
gradient_axis_mode = :total                 # Use :total or :avg_per_chain for the horizontal axis.

# Plots: these settings change the display, not sampling or metric evaluation.
shift_right = true                         # Include CMC initialisation costs on the horizontal axis.
line_stride_hmc = 25                       # Plot every nth CMC-HMC metric point.
line_stride_chmc = 10                     # Plot every nth CMC-CHMC metric point.
max_grad_evals = nothing                     # Right-hand axis cutoff; use nothing to show the full run.

# Run the experiment using the settings above.
BLAS.set_num_threads(blas_threads)
println("CPU threads: $(Threads.nthreads(:default)) available to Julia; " *
    "up to $chain_workers concurrent chains, $seed_workers concurrent seed initialisations; " *
    "BLAS threads: $(BLAS.get_num_threads()).")
seed_results = Dict{Symbol,Any}[]

# Prepare each seed's starting ensemble before running the production chains.
isempty(methods) && throw(ArgumentError("Select at least one CMC method."))
all(method -> method in (:cmc_hmc, :cmc_chmc_hmc), methods) ||
    throw(ArgumentError("Methods must be :cmc_hmc or :cmc_chmc_hmc."))
prepared_cmc_by_seed = prepare_cmc_initialisations(seeds, cfg;
    n_chains=n_chains,
    L=cfg.L,
    cmc_settings=cmc_settings,
    seed_workers=seed_workers,
    profile_timing=profile_timing,
    verbose=verbose,
)

# Show the energy fits before running the production chains.
mbar_fit_data = if !isempty(prepared_cmc_by_seed) && cmc_settings.display_mbar_plots
    [prepared_cmc_by_seed[seed].init.mbar_plot_data for seed in seeds]
else
    nothing
end

mbar_fit_row = mbar_fit_data === nothing ? nothing : plot_mbar_fit_row(mbar_fit_data, seeds; display_plot=false)
mbar_fit_row === nothing || display_plot_window(mbar_fit_row)

# Run the selected methods for every seed before combining the results.
for seed in seeds
    println("\n================ Seed $seed ================")
    results, _, _ = run_cmc_experiment(
        seed,
        cfg.target_name;
        target_id=target_id,
        dim=cfg.dim,
        pdb_name=cfg.pdb_name,
        target_kwargs=cfg.target_kwargs,
        n_chains=n_chains,
        chain_workers=chain_workers,
        n_samples=n_samples_hmc,
        L=cfg.L,
        methods=methods,
        oracle_size=oracle_size,
        oracle_ref_path=cfg.oracle_ref_path,
        ϵ_global=cfg.global_stepsize,
        method_settings=method_settings,
        cmc_settings=cmc_settings,
        prepared_cmc=pop!(prepared_cmc_by_seed, seed, nothing),
        metrics_n_projections=metrics_n_projections,
        metrics_reductions=metrics_reductions,
        pe_nbins=pe_nbins,
        gradient_axis_mode=gradient_axis_mode,
        profile_timing=profile_timing,
        verbose=verbose,
    )

    # The runner has already released each method's chain history; keep its metrics.
    push!(seed_results, results)
    println("\n$(length(seed_results))/$(length(seeds)) seeds completed")
end

# Combine all seeds and plot their convergence metrics.
results_avg = aggregate_over_seeds(seed_results)
plots = plot_results_summary(results_avg;
    display_plots=false,
    line_stride=line_stride_hmc,
    line_stride_cmc_chmc_hmc=line_stride_chmc,
    shift_right=shift_right,
    max_grad_evals=max_grad_evals,
)

# Open the convergence figures in separate windows.
display_plot_window(plots.projections)
display_plot_window(plots.pe)

# Report when the combined curves first reach the reference-error thresholds.
# The final figures remain available as plots.projections and plots.pe.
print_reference_band_hitting_times(results_avg; shift_right=shift_right)
