using Random
using JLD2

# Dictionary of settings for various benchmark targets.
const TARGET_SETTINGS = Dict{Symbol,NamedTuple}(
    :rosenbrock_36d => (  
        label = "Rosenbrock (36D)",
        target_name = :rosenbrock,
        dim = 36,
        target_kwargs = (
            mu = 1.0,
            sigma = 1.0,
            Q = 0.01,
        ),
        oracle_ref_file = "REF_DRAWS_rosenbrock_36d.jld2",
        global_stepsize = 0.025,
        L = 15,
        cmc_tuning = (h_chmc = 0.01, L_chmc = 50),
        u_min = 5.0,
        u_max = 75.0,
        sigma_u_denom = 16.0,
        K = 24,
        replica_exchange = true, 
        exchange_interval = 4,   
        nsteps = 6_000,
        burn = 1_000,
        L_umbrella = 10,
    ),
    :funnel_10d => (  
        label = "Neal's Funnel (10D)",
        target_name = :funnel,
        dim = 10,
        global_stepsize = 0.3,
        L = 15,
        cmc_tuning = (h_chmc = 0.05, L_chmc = 20),
        u_min = -40.0,
        u_max = 80.0,
        sigma_u_denom = 16.0,
        K = 24,
        replica_exchange = true, 
        exchange_interval = 4,   
        nsteps = 12_500,
        burn = 1_000,
        L_umbrella = 15,
    ),
    :funnel_100d => ( 
        label = "Neal's Funnel (100D)",
        target_name = :funnel,
        dim = 100,
        global_stepsize = 0.15,
        L = 30,
        cmc_tuning = (h_chmc = 0.02, L_chmc = 200),
        u_min = -500.0,
        u_max = 800.0,
        sigma_u_denom = 24.0,
        K = 37,
        replica_exchange = true,  
        exchange_interval = 4,  
        nsteps = 100_000,
        burn = 10_000,
        L_umbrella = 20,
    ),
    :lsat_centered => (
        label = "LSAT item response model, centred abilities (1006D)",
        target_name = :lsat_centered,
        dim = 1006,
        oracle_ref_file = "REF_DRAWS_lsat_centered.jld2",
        global_stepsize = 0.025,
        L = 125,
        cmc_tuning = (h_chmc = 0.01, L_chmc = 20),
        u_min = 2200.0,
        u_max = 2750.0,
        sigma_u_denom = 12.0,
        K = 19,
        nsteps = 10_000,
        burn = 2_000,
        L_umbrella = 50,
    ),
    :irt_2pl => (
        label = "Hierarchical two-parameter logistic IRT (144D)",
        target_name = :pdb,
        pdb_name = "irt_2pl-irt_2pl",
        dim = 144,
        oracle_ref_file = "REF_DRAWS_irt_2pl.jld2",
        global_stepsize = 0.025,
        L = 60,
        cmc_tuning = (h_chmc = 0.01, L_chmc = 20),
        u_min = 800.0,
        u_max = 1100.0,
        sigma_u_denom = 8.0,
        K = 13,
        nsteps = 5_000,
        burn = 1_000,
        L_umbrella = 20,
    ),
    :butterfly => (  
        label = "Butterfly multispecies occupancy model (106D)",
        target_name = :pdb,
        pdb_name = "butterfly-multi_occupancy",
        dim = 106,
        oracle_ref_file = "REF_DRAWS_butterfly.jld2",
        global_stepsize = 0.03,
        L = 120,
        cmc_tuning = (h_chmc = 0.01, L_chmc = 20),
        u_min = 1000.0,
        u_max = 1500.0,
        sigma_u_denom = 8.0,
        K = 13,
        nsteps = 5_000,
        burn = 1_000,
        L_umbrella = 20,
    ),
    :sblrc_blr => (
        label = "sblrc_blr",
        target_name = :pdb,
        pdb_name = "sblrc-blr",
        dim = 10,
        global_stepsize = 0.0004,
        L = 15,
        cmc_tuning = (h_chmc = 0.0001, L_chmc = 1000),
        u_min = 160.0,
        u_max = 190.0,
        sigma_u_denom = 12.0,
        K = 19,
        replica_exchange = true,
        exchange_interval = 4,
        nsteps = 10_000,
        burn = 1_000,
        L_umbrella = 15,
    ),
    :ark_ark => (
        label = "AR(5)",
        target_name = :pdb,
        pdb_name = "arK-arK",
        dim = 7,
        global_stepsize = 0.01,
        L = 30,
        cmc_tuning = (h_chmc = 0.0025, L_chmc = 50),
        u_min = -275.0,
        u_max = -250.0,
        sigma_u_denom = 6.0,
        K = 10,
        nsteps = 1_250,
        burn = 250,
        L_umbrella = 15,
    ),
    :arma_11 => (
        label = "ARMA(1,1)",
        target_name = :pdb,
        pdb_name = "arma-arma11",
        dim = 3,
        global_stepsize = 0.01,
        L = 10,
        cmc_tuning = (h_chmc = 0.0025, L_chmc = 100),
        u_min = -260.0,
        u_max = -240.0,
        sigma_u_denom = 6.0,
        K = 10,
        nsteps = 1_250,
        burn = 250,
        L_umbrella = 15,
    ),
    :eight_schools_centered => ( 
        label = "Eight Schools (Hierarchical Normal, Centred)",
        target_name = :pdb,
        pdb_name = "eight_schools-eight_schools_centered",
        dim = 10,
        global_stepsize = 0.2,
        L = 60,
        cmc_tuning = (h_chmc = 0.05, L_chmc = 20),
        u_min = -40.0,
        u_max = 50.0,
        sigma_u_denom = 20.0,
        K = 35,
        nsteps = 10_000,
        burn = 2_000,
        L_umbrella = 15,
    ),
    :lotka_volterra => (
        label = "Hudson Lynx-Hare (Lotka-Volterra)",
        target_name = :pdb,
        pdb_name = "hudson_lynx_hare-lotka_volterra",
        dim = 8,
        global_stepsize = 0.015,
        L = 30,
        cmc_tuning = (h_chmc = 0.0025, L_chmc = 400),
        u_min = 9.626376,
        u_max = 40.0,
        sigma_u_denom = 8.0,
        K = 13,
        nsteps = 2_500,
        burn = 500,
        L_umbrella = 15,
    ),
    :one_comp => (
        label = "One Compartment PK (Hierarchical Normal)",
        target_name = :pdb,
        pdb_name = "one_comp_mm_elim_abs-one_comp_mm_elim_abs",
        dim = 4,
        global_stepsize = 0.08,
        L = 10,
        cmc_tuning = (h_chmc = 0.025, L_chmc = 20),
        u_min = -11,
        u_max = 10,
        sigma_u_denom = 8.0,
        K = 13,
        nsteps = 2_500,
        burn = 500,
        L_umbrella = 10,
    ),
    :german_credit_full => (
        label = "German Credit (302D, 100 obs.)",
        target_name = :german_credit,
        dim = 302,
        target_kwargs = (num_obs = 100, num_latent = 300, no_pca = true),
        needs_datadir = true,
        oracle_ref_file = "REF_DRAWS_german_credit_302d_100.jld2",
        global_stepsize = 0.05,
        L = 60,
        cmc_tuning = (h_chmc = 0.05, L_chmc = 20),
        u_min = -300.0,
        u_max = 900.0,
        sigma_u_denom = 20.0,
        K = 35,
        replica_exchange = true,  
        exchange_interval = 4,  
        nsteps = 10_000,
        burn = 1_000,
        L_umbrella = 60,
    ),
    :finnish_pines_8 => (
        label = "Finnish Pines (LGCPP, 8x8)",
        target_name = :finnish_pines,
        dim = 64,
        target_kwargs = (ngrid = 8,),
        needs_datadir = true,
        oracle_ref_file = "REF_DRAWS_finnish_pines.jld2",
        global_stepsize = 0.225,
        L = 15,
        cmc_tuning = (h_chmc = 0.15, L_chmc = 15),
        u_min = -450.0,
        u_max = -360.0,
        sigma_u_denom = 12.0,
        K = 19,
        nsteps = 4_000,
        burn = 1_000,
        L_umbrella = 15,
    ),
)

# Helper to read key from a NamedTuple, else return a default value
get_key_from_nt(nt::NamedTuple, key::Symbol, default) = hasproperty(nt, key) ? getproperty(nt, key) : default

# Return a sorted list of available target setting IDs
available_target_settings() = sort!(collect(keys(TARGET_SETTINGS)))

# Given a target setting ID, return a NamedTuple with all the relevant settings for that target.
function target_setting(id::Symbol; target_datadir::AbstractString = joinpath(@__DIR__, "data"),
    ref_draws_dir::AbstractString = joinpath(@__DIR__, "ref_draws"))

    haskey(TARGET_SETTINGS, id) || error("Unknown target setting $id. Available: $(available_target_settings())")
    row = TARGET_SETTINGS[id]

    target_kwargs = get_key_from_nt(row, :target_kwargs, NamedTuple())
    if get_key_from_nt(row, :needs_datadir, false)
        target_kwargs = merge((datadir = target_datadir,), target_kwargs)
    end

    oracle_ref_file = get_key_from_nt(row, :oracle_ref_file, nothing)
    oracle_ref_path = oracle_ref_file === nothing ? nothing : joinpath(ref_draws_dir, oracle_ref_file)

    row.sigma_u_denom > 0 || error("sigma_u_denom must be positive for $id")
    row.K >= 2 || error("K must be at least 2 for $id")

    exchange_interval = get_key_from_nt(row, :exchange_interval, 16)
    exchange_interval isa Integer && !(exchange_interval isa Bool) && exchange_interval > 0 ||
        throw(ArgumentError("exchange_interval must be a positive integer for $id"))
    replica_exchange = get_key_from_nt(row, :replica_exchange, false)
    replica_exchange isa Bool || throw(ArgumentError("replica_exchange must be true or false for $id"))

    umbrella = (
        u_min = Float64(row.u_min),
        u_max = Float64(row.u_max),
        σu = Float64((row.u_max - row.u_min) / row.sigma_u_denom),
        K = Int(row.K),
        replica_exchange = replica_exchange,
        exchange_interval = Int(exchange_interval),
        nsteps = Int(row.nsteps),
        burn = Int(row.burn),
        eps_init_umbrella = Float64(row.global_stepsize),
        L_umbrella = Int(row.L_umbrella),
    )
    cmc_tuning_row = row.cmc_tuning
    cmc_tuning = (
        h_chmc = Float64(cmc_tuning_row.h_chmc),
        L_chmc = Int(cmc_tuning_row.L_chmc),
    )

    return (
        id = id,
        label = row.label,
        target_name = row.target_name,
        dim = Int(row.dim),
        pdb_name = get_key_from_nt(row, :pdb_name, ""),
        target_kwargs = target_kwargs,
        oracle_ref_path = oracle_ref_path,
        global_stepsize = Float64(row.global_stepsize),
        L = Int(row.L),
        cmc_tuning = cmc_tuning,
        umbrella = umbrella,
    )
end
# Given a NamedTuple of target settings, build the corresponding target distribution object.
function build_target(cfg::NamedTuple)
    return make_target(cfg.target_name; dim = cfg.dim, pdb_name = cfg.pdb_name, cfg.target_kwargs...)
end

# Loads/generates reference samples for a given target.
function load_oracle_samples(rng::AbstractRNG, ℓπ, cfg::NamedTuple; n_samples::Int)
    if cfg.target_name == :funnel
        return funnel_oracle(rng, ℓπ.t, n_samples)

    elseif cfg.target_name == :pdb
        ref_path = cfg.oracle_ref_path
        if ref_path !== nothing
            isfile(ref_path) || error("Missing reference draws file for $(cfg.pdb_name): $ref_path")
            return JLD2.jldopen(ref_path, "r"; parallel_read=true) do file
                file["X_ref"]
            end
        end
        return posteriordb_oracle_unconstrained(rng, ℓπ.t, n_samples)
    elseif cfg.target_name == :lsat_centered
        cfg.oracle_ref_path === nothing && error("oracle_ref_path is required for centred LSAT")
        isfile(cfg.oracle_ref_path) || error("Missing centred LSAT reference: $(cfg.oracle_ref_path).")
        return JLD2.jldopen(cfg.oracle_ref_path, "r"; parallel_read=true) do file
            haskey(file, "diag") || error("Centred LSAT reference requires coordinate metadata.")
            diag = file["diag"]
            get(diag, :model_spec, nothing) == LSAT_CENTERED_SPEC &&
                get(diag, :draw_coordinates, nothing) == :centered ||
                error("Reference is not in centred LSAT coordinates.")
            X = file["X_ref"]
            size(X, 1) == MCMCLogDensityProblems.dim(ℓπ.t) ||
                error("Centred LSAT reference dimension mismatch.")
            return X
        end
    elseif cfg.target_name in (:german_credit, :finnish_pines, :rosenbrock)
        cfg.oracle_ref_path === nothing && error("oracle_ref_path is required for $(cfg.target_name)")
        isfile(cfg.oracle_ref_path) || error("Missing reference draws for $(cfg.target_name): $(cfg.oracle_ref_path). Generate references for this target before comparing samplers.")
        return JLD2.jldopen(cfg.oracle_ref_path, "r"; parallel_read=true) do file
            file["X_ref"]
        end
    end
    error("Unknown target_name $(cfg.target_name) in load_oracle_samples.")
end
