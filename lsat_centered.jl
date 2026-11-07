# Adds support for the centred LSAT target.
struct CenteredLSAT{T}
    base::T
    ability_indices::Vector{Int}
    logbeta_index::Int
end

const LSAT_CENTERED_SPEC = :lsat_beta_scaled_abilities_v1

function CenteredLSAT(base)
    base.posterior_name == "lsat_data-lsat_model" ||
        throw(ArgumentError("CenteredLSAT requires lsat_data-lsat_model"))
    names = BridgeStan.param_unc_names(base.bsmodel)
    abilities = findall(n -> startswith(n, "theta."), names)
    beta_index = findfirst(==("beta"), names)
    length(abilities) == 1000 && beta_index !== nothing && length(names) == 1006 ||
        error("Unexpected LSAT parameter layout; cannot safely transform abilities.")
    return CenteredLSAT(base, abilities, beta_index)
end

make_lsat_centered_target(; kwargs...) =
    CenteredLSAT(make_posteriordb_target("lsat_data-lsat_model"; kwargs...))
MCMCLogDensityProblems.dim(t::CenteredLSAT) = MCMCLogDensityProblems.dim(t.base)
check_chain_target(t::CenteredLSAT, workers::Int) = check_chain_target(t.base, workers)

function lsat_to_original!(dest::AbstractVector, t::CenteredLSAT,
    q::AbstractVector)
    length(dest) == length(q) == MCMCLogDensityProblems.dim(t) ||
        throw(DimensionMismatch("Invalid LSAT draw dimension"))
    scale = exp(-q[t.logbeta_index])
    copyto!(dest, q)
    @inbounds for j in t.ability_indices
        dest[j] = q[j] * scale
    end
    return dest
end

function target_logpdf(t::CenteredLSAT, q::AbstractVector{<:Real})
    original = lsat_to_original!(Vector{Float64}(undef, length(q)), t, q)
    all(isfinite, original) || return -Inf
    lp = target_logpdf(t.base, original) - length(t.ability_indices)*q[t.logbeta_index]
    return isfinite(lp) ? lp : -Inf
end

function MCMCLogDensityProblems.logpdf_grad(t::CenteredLSAT, q::AbstractVector)
    original = lsat_to_original!(Vector{Float64}(undef, length(q)), t, q)
    all(isfinite, original) || return (-Inf, zeros(length(q)))
    lp, g = MCMCLogDensityProblems.logpdf_grad(t.base, original)
    k = t.logbeta_index
    lp -= length(t.ability_indices)*q[k]
    correction = 0.0
    invbeta = exp(-q[k])
    @inbounds for j in t.ability_indices
        correction += g[j]*original[j]
        g[j] *= invbeta
    end
    g[k] -= correction + length(t.ability_indices)
    if !isfinite(lp) || !all(isfinite, g)
        return -Inf, zeros(length(q))
    end
    return lp, g
end
