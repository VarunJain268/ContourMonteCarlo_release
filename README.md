# Contour Monte Carlo

This repository provides a multithreaded CPU implementation of Contour Monte Carlo (CMC) for the supplied benchmark targets. It runs independent seed initialisations and sampling chains in parallel, then compares the samples with reference draws. Start with [main.jl](main.jl): its settings block controls the target, seeds, methods, chain counts, metrics and plots. Target-specific defaults live in [target_settings.jl](target_settings.jl).

Use Julia 1.12; the recorded environment was tested with Julia 1.12.4. Open a terminal (e.g., the VS Code terminal) in this repository directory. Run these commands in the terminal, not in the Julia REPL. Install the packages once:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Edit the settings in `main.jl`, then run:

```bash
julia --project=. --threads=auto -i main.jl
```

`--threads=auto` automatically sets the worker thread count to the number of logical CPUs available to Julia; `-i` leaves an interactive Julia prompt after the run.

**Note:** Memory use grows with the number of chains, samples per chain and target dimension. Large runs can therefore be memory-intensive, especially for high-dimensional targets.

The experiment follows these steps:

1. **Build the target.** The selected preset supplies the model, sampler settings and reference-file location. References come from `ref_draws/`, PosteriorDB, or exact sampling for the funnel. Model data are kept in `data/`. PosteriorDB models use BridgeStan and need a working C++ build toolchain for their first compilation.
2. **Prepare starting samples.** Short pilot chains provide starting positions for energy-biased umbrella windows. The windows run independently or exchange states, according to the preset. MBAR combines their samples into a weighted discrete approximation, then resamples positions and their energies to initialise CMC. Histogram bins are used for plots and energy-KL evaluation, not for these discrete MBAR weights.
3. **Run the selected methods.** `:cmc_hmc` runs HMC from those starting positions. `:cmc_chmc_hmc` first runs CHMC on each chain's energy contour, then optionally continues with HMC. Its `alpha` setting controls the CHMC fraction; `alpha = 1.0` means CHMC only. Both methods share the initialisation for a given seed.
4. **Evaluate convergence.** At each checkpoint, samples across chains are compared with reference draws using KS, CDF-ISE and histogram-based energy KL. Plots show median and worst-case projection errors against target gradient evaluations.
5. **Combine seeds and plot once.** After every seed finishes, metrics are combined into median curves with interquartile bands. The energy-fit and convergence plots open in separate windows at the end. Results remain in memory as `results_avg`, `plots.projections`, `plots.pe` and `mbar_fit_row`; files are not saved automatically.

The main implementation is organised as follows:

| Files | Purpose |
| --- | --- |
| `cmc.jl` | Connects initialisation, sampling and metric evaluation. |
| `umbrella.jl`, `mbar.jl`, `umbrella_mbar_pe_estimation.jl` | Pilot sampling, umbrella windows, discrete MBAR weights and starting draws. |
| `chains.jl`, `chmc.jl`, `chain_execution.jl` | HMC/CHMC transitions, parallel chains and gradient accounting. |
| `targets.jl`, `pdb_interface.jl`, `lsat_centered.jl` | Target densities and model interfaces. |
| `distances.jl`, `convergence_metrics.jl`, `seed_aggregation.jl`, `plotting.jl` | Error metrics, reference baselines, seed aggregation and figures. |

[tuning.jl](tuning.jl) is a separate standalone file for tuning HMC step sizes and leapfrog counts using NUTS pilots. [exact.jl](exact.jl) contains separate exact-sampling examples for radial targets and is not part of the main CMC workflow.
