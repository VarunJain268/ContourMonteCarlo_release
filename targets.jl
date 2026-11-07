# This file defines a collection of target distributions
using MCMCLogDensityProblems
using LogDensityProblems
using Random
using Distributions
using LinearAlgebra
using Statistics
using BSON
using StatsFuns: logistic, softplus
using MultivariateStats

const LOG2π = log(2π)

include(joinpath(@__DIR__, "lsat_centered.jl"))

# General-d Neal's Funnel: 1 scale variable ν + (dim-1) spokes x
struct FunnelND <: ContinuousMultivariateDistribution
    dim::Int
    σν::Float64
    μ::Vector{Float64}   # translation offset
end

function FunnelND(; dim::Int, σν::Float64=3.0, μ::AbstractVector{<:Real}=zeros(dim))
    length(μ) == dim || throw(ArgumentError("μ must have length $dim (got $(length(μ)))"))
    return FunnelND(dim, σν, Float64.(μ))
end

dim(f::FunnelND) = f.dim

function target_logpdf(f::FunnelND, θ::AbstractVector{<:Real})
    d = length(θ); @assert d == f.dim && d ≥ 2
    m = d - 1
    μ = f.μ

    v = θ[1] - μ[1]

    sumsq = 0.0
    @inbounds for i in 1:m
        dx = θ[i+1] - μ[i+1]
        sumsq += dx * dx
    end

    invs2 = exp(-v)

    return (-0.5 * LOG2π - log(f.σν) - 0.5 * (v / f.σν)^2) +
           (-0.5 * m * LOG2π - 0.5 * m * v - 0.5 * invs2 * sumsq)
end

function logpdf_grad(f::FunnelND, θ::AbstractVector)
    lp = target_logpdf(f, θ)

    # gradients
    g = similar(θ, Float64)
    m = length(θ) - 1
    μ = f.μ
    v = θ[1] - μ[1]
    sumsq = 0.0
    @inbounds for i in 1:m
        dx = θ[i+1] - μ[i+1]
        sumsq += dx * dx
    end
    invs2 = exp(-v)

    g[1] = -(v / (f.σν^2)) - m / 2 + 0.5 * invs2 * sumsq
    @inbounds for i in 1:m
        dx = θ[i+1] - μ[i+1]
        g[i+1] = -invs2 * dx
    end
    return lp, g
end

MCMCLogDensityProblems.dim(f::FunnelND) = f.dim
MCMCLogDensityProblems.logpdf_grad(f::FunnelND, θ::AbstractVector) = logpdf_grad(f, θ)

# German credit
function reduce_dim_by_pca(design_matrix, target_dim::Int)
    M = fit(MultivariateStats.PCA, design_matrix'; maxoutdim=target_dim)
    return Matrix(MultivariateStats.predict(M, design_matrix')')
end

function german_credit_target(; 
    lambda::Float64,  ## This is rate we pass in = 1/scale
    datadir::Union{Nothing,String},
    num_obs::Int,
    num_latent::Int,
    no_pca::Bool,
)
    dd = datadir
    bson_path = joinpath(dd, "germancredit.bson")
    isfile(bson_path) || error("Missing data file: $bson_path")

    d = BSON.load(bson_path)                     # Dict
    X = d[:design_matrix]
    y = Bool.(d[:response])

    println(size(X))  

    if size(X, 2) != num_latent
        X = no_pca ? X[:, 1:num_latent] : reduce_dim_by_pca(X, num_latent)
    end
    X = X[1:num_obs, :]
    y = y[1:num_obs]

    # Use the struct constructor, which is (X, y, lambda)
    return MCMCLogDensityProblems.LogisticRegression(X, y, lambda)
end

function centered_logistic_evaluate(lr::MCMCLogDensityProblems.LogisticRegression,
        θ::AbstractVector; gradient::Bool=true)
    X = lr.X
    yb = lr.y              
    N, p = size(X)
    @assert length(θ) == p + 2

    logv = θ[1]
    v    = exp(logv)
    a    = θ[2]
    b    = @view θ[3:end]

    η = X * b
    @inbounds @simd for i in eachindex(η)
        η[i] += a
    end

    r = gradient ? similar(η) : nothing
    ll = 0.0
    @inbounds for i in 1:N
        yi = ifelse(yb[i], 1.0, 0.0)
        gradient && (r[i] = yi - logistic(η[i]))
        ll += yi*η[i] - softplus(η[i])
    end

    q = a*a + dot(b, b) 
    lp_v = log(lr.lambda) - lr.lambda*v + logv
    lp_b = -0.5 * ((p+1)*(LOG2π + log(v)) + q / v)

    logp = ll + lp_v + lp_b
    gradient || return logp, nothing

    ga = sum(r) - a / v
    gb = X' * r .- b / v
    g_logv = 1 - lr.lambda*v - 0.5*(p+1) + 0.5*q / v

    g = similar(θ)
    g[1] = g_logv
    g[2] = ga
    g[3:end] = gb
    return logp, g
end

logpdf_grad(lr::MCMCLogDensityProblems.LogisticRegression, θ::AbstractVector) =
    centered_logistic_evaluate(lr, θ)
target_logpdf(lr::MCMCLogDensityProblems.LogisticRegression, θ::AbstractVector{<:Real}) =
    first(centered_logistic_evaluate(lr, θ; gradient=false))

MCMCLogDensityProblems.dim(lr::LogisticRegression) = size(lr.X, 2) + 2
MCMCLogDensityProblems.logpdf_grad(lr::LogisticRegression, θ::AbstractVector) =
    logpdf_grad(lr, θ)

# Log-Gaussian Cox point process
function cox_process_target(; datadir::String, ngrid::Int)
    path = joinpath(datadir, "finpines-$ngrid.bson")
    isfile(path) || error("Missing data file: $path")
    println("Found data file $path")

    d = BSON.load(path)  
    getk(sym) = haskey(d, sym) ? d[sym] : d[string(sym)]

    counts  = Vector{Int}(getk(:data_counts))
    dim_in  = Int(getk(:dimension))
    sigmasq = Float64(getk(:sigmasq))
    mu      = Float64(getk(:mu))
    beta    = Float64(getk(:beta))
    area    = Float64(getk(:area))

    length(counts) == dim_in || error("length(counts)=$(length(counts)) ≠ dimension=$dim_in")
    dim_in == ngrid^2      || error("dimension=$dim_in but ngrid=$ngrid ⇒ ngrid^2=$(ngrid^2)")

    μ = fill(mu, dim_in)
    Σ = Matrix{Float64}(undef, dim_in, dim_in)

    for m in 1:dim_in, n in 1:dim_in
        i = [floor(Int, (m - 1) / ngrid) + 1, (m - 1) % ngrid + 1]
        j = [floor(Int, (n - 1) / ngrid) + 1, (n - 1) % ngrid + 1]
        Σ[m,n] = sigmasq * exp(-sqrt(sum((i - j).^2)) / (ngrid * beta))
    end
    prior_X = MvNormal(μ, Σ)
    return MCMCLogDensityProblems.LogGaussianCoxPointProcess(counts, dim_in, area, prior_X)
end

function logpdf_grad(pines::MCMCLogDensityProblems.LogGaussianCoxPointProcess,
                     θ::AbstractVector)
    @assert length(θ) == pines.dimension
    μ  = mean(pines.prior_X)               
    Σ  = cov(pines.prior_X)                
    y  = pines.counts
    a  = pines.area

    ex = exp.(θ)
    ll = dot(θ, y) - a * sum(ex)

    lp = logpdf(pines.prior_X, θ)
    logp = lp + ll

    g = similar(θ, promote_type(eltype(θ), Float64))
    @. g = y - a * ex
    g .-= (Σ \ (θ - μ))

    return logp, g
end

MCMCLogDensityProblems.dim(pines::MCMCLogDensityProblems.LogGaussianCoxPointProcess) = pines.dimension
MCMCLogDensityProblems.logpdf_grad(pines::MCMCLogDensityProblems.LogGaussianCoxPointProcess, θ::AbstractVector) =
    logpdf_grad(pines, θ)

# Rosenbrock
struct RosenbrockTarget <: ContinuousMultivariateDistribution
    dim::Int
    mu::Float64
    sigma::Float64
    Q::Float64
end

function RosenbrockTarget(; dim::Int, mu::Real, sigma::Real, Q::Real)
    dim >= 2 || throw(ArgumentError("dim must be at least 2."))
    sigma > 0 || throw(ArgumentError("sigma must be positive."))
    Q > 0 || throw(ArgumentError("Q must be positive."))
    return RosenbrockTarget(dim, Float64(mu), Float64(sigma), Float64(Q))
end

MCMCLogDensityProblems.dim(t::RosenbrockTarget) = t.dim
Base.length(t::RosenbrockTarget) = t.dim

function MCMCLogDensityProblems.logpdf(t::RosenbrockTarget, θ::AbstractVector)
    length(θ) == t.dim || throw(ArgumentError("θ has length $(length(θ)) but dim=$(t.dim)"))
    all(isfinite, θ) || return -Inf

    sigma2 = t.sigma^2
    Q = t.Q

    U = 0.0

    @inbounds for i in 1:(t.dim - 1)
        x = θ[i]
        y = θ[i + 1]

        r = y - x^2
        a = x - t.mu

        U += 0.5 * (r^2 / Q + a^2 / sigma2)
    end

    return -U
end

function MCMCLogDensityProblems.logpdf_grad(t::RosenbrockTarget, θ::AbstractVector)
    length(θ) == t.dim || throw(ArgumentError("θ has length $(length(θ)) but dim=$(t.dim)"))

    if !all(isfinite, θ)
        return -Inf, zeros(Float64, t.dim)
    end

    sigma2 = t.sigma^2
    Q = t.Q

    U = 0.0
    g = zeros(Float64, t.dim)

    @inbounds for i in 1:(t.dim - 1)
        x = θ[i]
        y = θ[i + 1]

        r = y - x^2
        a = x - t.mu

        U += 0.5 * (r^2 / Q + a^2 / sigma2)
        g[i]     +=  2.0 * x * r / Q - a / sigma2
        g[i + 1] += -r / Q
    end

    logp = -U

    if !(isfinite(logp) && all(isfinite, g))
        return -Inf, zeros(Float64, t.dim)
    end
    return logp, g
end

# Factory function to create targets
function make_target(name::Symbol; dim::Int, pdb_name::String, kwargs...)
    kw = Dict(kwargs)
    t = name == :rosenbrock ? RosenbrockTarget(dim=dim; kwargs...) :
        name == :funnel  ? FunnelND(dim=dim; kwargs...) :
        name == :pdb     ? make_posteriordb_target(pdb_name; kwargs...) :
        name == :lsat_centered ? make_lsat_centered_target(; kwargs...) :
        name === :german_credit ? german_credit_target(; 
            lambda = get(kw, :lambda, 1.0),
            datadir = haskey(kw, :datadir) ? kw[:datadir] : error("Must provide datadir for :german_credit"),
            num_obs = get(kw, :num_obs, 1000),
            num_latent = get(kw, :num_latent, 300),
            no_pca = get(kw, :no_pca, false),
        ) :
        name == :finnish_pines ? cox_process_target(; 
            datadir = haskey(kw, :datadir) ? kw[:datadir] : error("Must provide datadir for :finnish_pines"),
            ngrid = get(kw, :ngrid, 8),
        ) :
        error("Unknown target name: $name. Use :rosenbrock, :funnel, :pdb, :lsat_centered, :german_credit, :finnish_pines.")
    return LDP(t)
end

# Exact i.i.d. sampling oracle for FunnelND.
function funnel_oracle(rng::AbstractRNG, f::FunnelND, n::Integer)
    n ≥ 1 || throw(ArgumentError("n must be ≥ 1 (got $n)"))
    d = f.dim
    d ≥ 2 || throw(ArgumentError("dim must be ≥ 2 (got $d)"))

    X = Matrix{Float64}(undef, d, n)

    @inbounds for j in 1:n
        # scale variable v ~ Normal(0, σν^2)
        v = f.σν * randn(rng)
        X[1, j] = v + f.μ[1]

        # spokes: x_i | v ~ Normal(0, exp(v)) independently
        s = exp(0.5 * v)  # std = sqrt(exp(v))
        for i in 2:d
            X[i, j] = s * randn(rng) + f.μ[i]
        end
    end

    return X
end
