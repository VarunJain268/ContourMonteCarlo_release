# Provide the target interface, reproducible random streams, and shared numerical helpers.

if !@isdefined run_chain_jobs
    include(joinpath(@__DIR__, "chain_execution.jl"))
end

using LogDensityProblems
using MCMCLogDensityProblems
using LinearAlgebra
using Random
using Statistics

struct LDP{T}
    t::T
end

check_chain_target(target::LDP, workers::Int) = check_chain_target(target.t, workers)


const EXPERIMENT_RNG_STREAM_IDS = Dict{Symbol,UInt64}(
    :cmc_initialisation => 0x08,
    :cmc_mbar => 0x09,
    :cmc_hmc => 0x0a,
    :cmc_chmc_hmc => 0x0b,
)

@inline function splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function experiment_stream_seed(seed::Integer, stream::Symbol)
    stream_id = get(EXPERIMENT_RNG_STREAM_IDS, stream, nothing)
    stream_id === nothing &&
        throw(ArgumentError("Unknown experiment RNG stream :$stream."))
    seed_bits = reinterpret(UInt64, Int64(seed))
    mixed = splitmix64(seed_bits ⊻ (stream_id * 0x9e3779b97f4a7c15))
    return Int(mixed & UInt64(typemax(Int)))
end

experiment_rng(seed::Integer, stream::Symbol) =
    MersenneTwister(experiment_stream_seed(seed, stream))

@inline target_logpdf(t, θ::AbstractVector{<:Real}) = MCMCLogDensityProblems.logpdf(t, θ)
@inline function target_logpdf_grad(t, θ::AbstractVector{<:Real})
    increment_target_grad_counter!()
    return MCMCLogDensityProblems.logpdf_grad(t, θ)
end


LogDensityProblems.logdensity_and_gradient(ℓ::LDP, θ::AbstractVector{<:Real}) = target_logpdf_grad(ℓ.t, θ)
LogDensityProblems.dimension(ℓ::LDP) = MCMCLogDensityProblems.dim(ℓ.t)
LogDensityProblems.capabilities(::Type{<:LDP}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(ℓ::LDP, θ::AbstractVector{<:Real}) = target_logpdf(ℓ.t, θ)

# Other helpers 
logπ_and_∇logπ(ℓ::LDP, q::AbstractVector{<:Real}) = target_logpdf_grad(ℓ.t, q)
U_and_∇U(ℓ::LDP, q::AbstractVector) = begin
    v, g = logπ_and_∇logπ(ℓ, q)
    return -v, -g
end

@inline finite_vec(x) = Float64.(filter(isfinite, collect(skipmissing(x)))) # Drop missing, non-finite values and convert to Float64

mean_finite(v) = begin
    fv = finite_vec(v)
    isempty(fv) ? NaN : mean(fv)
end

function summary_stats(x::AbstractVector{<:Real})
    n = length(x)
    n == 0 && return "n=0"
    xf = finite_vec(x)
    isempty(xf) && return "n=$n, finite=0"
    s = (min=minimum(xf), median=median(xf), max=maximum(xf), mean=mean(xf))
    return "n=$n, finite=$(length(xf)), mean=$(round(s.mean, digits=6)), median=$(round(s.median, digits=6)), min=$(round(s.min, digits=6)), max=$(round(s.max, digits=6))"
end

# Sampling helpers
function random_unit_vectors(rng::AbstractRNG, D::Int, K::Int)
    V = randn(rng, D, K)
    for k in 1:K
        nk = norm(@view V[:, k])
        nk > 0 || error("Zero vector encountered in random_unit_vectors.")
        @view(V[:, k]) ./= nk
    end
    return V
end
