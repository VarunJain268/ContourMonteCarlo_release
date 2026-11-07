# Code to interface with PosteriorDB  
using MCMCLogDensityProblems
using Random
using PosteriorDB
using BridgeStan
using SHA
using FileWatching: Pidfile

struct PosteriorDBTarget
    posterior_name::String
    bsmodel
    pdb
end

MCMCLogDensityProblems.dim(t::PosteriorDBTarget) = BridgeStan.param_unc_num(t.bsmodel)

function check_chain_target(t::PosteriorDBTarget, chain_workers::Int)
    if chain_workers > 1 && !occursin(r"STAN_THREADS\s*=\s*(true|1)"i, BridgeStan.model_info(t.bsmodel))
        throw(ArgumentError("Parallel PosteriorDB chains require a BridgeStan model compiled with STAN_THREADS=true; use chain_workers=1 for this model."))
    end
    return nothing
end

function safe_log_density(bsmodel, θ::AbstractVector)
    θv = θ isa Vector{Float64} ? θ : Vector{Float64}(θ)
    try
        return BridgeStan.log_density(bsmodel, θv)
    catch
        return -Inf
    end
end

function safe_log_density_gradient(bsmodel, θ::AbstractVector)
    θv = θ isa Vector{Float64} ? θ : Vector{Float64}(θ)
    try
        return BridgeStan.log_density_gradient(bsmodel, θv)
    catch
        return -Inf, fill(NaN, length(θv))
    end
end

function target_logpdf(t::PosteriorDBTarget, θ::AbstractVector{<:Real})
    @assert length(θ) == MCMCLogDensityProblems.dim(t)
    return safe_log_density(t.bsmodel, θ)
end

function MCMCLogDensityProblems.logpdf_grad(t::PosteriorDBTarget, θ::AbstractVector)
    @assert length(θ) == MCMCLogDensityProblems.dim(t)
    lp, g = safe_log_density_gradient(t.bsmodel, θ)
    if !isfinite(lp) || any(x -> !isfinite(x), g)
        return -Inf, zeros(Float64, length(θ))
    end
    return lp, g
end

function bridgestan_to_draw_key(name::AbstractString)
    parts = split(String(name), ".")
    length(parts) == 1 && return parts[1]
    return string(parts[1], "[", join(parts[2:end], ","), "]")
end

function lookup_draw_value(source, key::AbstractString)
    haskey(source, key) && return source[key]
    skey = Symbol(key)
    haskey(source, skey) && return source[skey]
    error("Reference draw is missing parameter key: $key")
end

function flatten_chain_draw(chain, draw_idx::Int, constrained_names::AbstractVector{<:AbstractString})
    θc = Vector{Float64}(undef, length(constrained_names))
    @inbounds for (i, cname) in enumerate(constrained_names)
        draw_key = bridgestan_to_draw_key(cname)
        v = lookup_draw_value(chain, draw_key)
        θc[i] = Float64(v isa AbstractVector ? v[draw_idx] : v)
    end
    return θc
end

function pdb_threaded_make_args(make_args::AbstractVector{String})
    args = String[]
    for arg in make_args
        assignment = match(r"^\s*STAN_THREADS\s*[:+?]?=\s*(.*?)\s*$", arg)
        if assignment === nothing
            push!(args, arg)
        else
            assignment.captures[1] in ("true", "1") || throw(ArgumentError(
                "PosteriorDB targets require STAN_THREADS=true; use chain_workers=1 for serial execution."))
        end
    end
    push!(args, "STAN_THREADS=true")
    return args
end

function make_posteriordb_target(posterior_name::String;
    make_args::AbstractVector{String}=String[],
    stanc_args::AbstractVector{String}=String[], bridgestan_kwargs...)
    isempty(posterior_name) && error("posterior_name must be non-empty for PosteriorDB targets.")
    threaded_make_args = pdb_threaded_make_args(make_args)
    pdb = PosteriorDB.database()
    post = PosteriorDB.posterior(pdb, posterior_name)
    model = PosteriorDB.model(post)
    impl = PosteriorDB.implementation(model, "stan")
    stan_path = PosteriorDB.path(impl)
    data_str = PosteriorDB.load(PosteriorDB.dataset(post), String)

    source = read(stan_path, String)

    bridge_root = realpath(BridgeStan.get_bridgestan_path())
    local_config = joinpath(bridge_root, "make", "local")
    build_config = isfile(local_config) ? read(local_config, String) : ""
    cache_key = bytes2hex(sha256(repr((source, threaded_make_args, stanc_args,
        string(Base.pkgversion(BridgeStan)), Sys.MACHINE, bridge_root, build_config))))
    cache_dir = joinpath(@__DIR__, ".pdb_cache", string(Sys.ARCH), "threaded", cache_key)
    mkpath(cache_dir)
    local_stan = joinpath(cache_dir, basename(stan_path))
    if !isfile(local_stan) || read(local_stan, String) != source
        write(local_stan, source)
    end

    bs = Pidfile.mkpidlock(joinpath(bridge_root, ".cmc-build.pid"); stale_age=3600) do
        BridgeStan.StanModel(local_stan, data_str;
            make_args=threaded_make_args, stanc_args=stanc_args, bridgestan_kwargs...)
    end
    target = PosteriorDBTarget(posterior_name, bs, pdb)
    check_chain_target(target, 2)  
    return target
end

function posteriordb_oracle_unconstrained(rng::AbstractRNG, t::PosteriorDBTarget, n::Int)
    post = PosteriorDB.posterior(t.pdb, t.posterior_name)
    rp = PosteriorDB.reference_posterior(post)
    isnothing(rp) && error("PosteriorDB reference draws not available for posterior: $(t.posterior_name)")

    chains = PosteriorDB.load(rp)
    chains isa AbstractVector || error("Expected chain-wise reference draws for $(t.posterior_name).")
    isempty(chains) && error("No reference draws available for posterior: $(t.posterior_name)")

    first_chain = chains[1]
    first_key = first(keys(first_chain))
    draws_per_chain = length(lookup_draw_value(first_chain, string(first_key)))
    n_chains = length(chains)
    n_avail = n_chains * draws_per_chain
    n_use = min(n, n_avail)
    flat_idx = n_use == n_avail ? collect(1:n_use) : randperm(rng, n_avail)[1:n_use]

    constrained_names = BridgeStan.param_names(t.bsmodel; include_tp=false, include_gq=false)
    d = BridgeStan.param_unc_num(t.bsmodel)
    X = Matrix{Float64}(undef, d, n_use)

    @inbounds for (k, j) in enumerate(flat_idx)
        chain_idx = div(j - 1, draws_per_chain) + 1
        draw_idx = (j - 1) % draws_per_chain + 1
        θc = flatten_chain_draw(chains[chain_idx], draw_idx, constrained_names)
        θu = BridgeStan.param_unconstrain(t.bsmodel, θc)
        X[:, k] .= θu
    end

    return X
end
