# Generate exact samples from radial targets and plot their distributions and marginals.

using Random
using LinearAlgebra
using Statistics
using Distributions
using KernelDensity
using QuadGK
using Plots
using LaTeXStrings
using Measures

gr(dpi = 600)

asvec(x) = Float64.(collect(x))

function invsqrt_from_scatter(Σ::AbstractMatrix)
    # Returns A such that A' * A = inv(Σ).
    # If Σ = L L', then A = inv(L).
    F = cholesky(Symmetric(Matrix{Float64}(Σ)))
    return inv(F.L)
end

function rand_betaprime(rng::AbstractRNG, a::Real, b::Real)
    # If X ~ Gamma(a,1), Y ~ Gamma(b,1), then X/Y ~ BetaPrime(a,b).
    return rand(rng, Gamma(a, 1.0)) / rand(rng, Gamma(b, 1.0))
end

struct RadialTarget{F,G}
    mu::Vector{Float64}
    A::Matrix{Float64}
    p::Float64
    c::Vector{Float64}
    psi::F
    sample_R::G
    name::String
end

function make_target(; mu, A, p::Real=2.0, c=nothing, psi, sample_R, name::String="custom")
    mu = asvec(mu)
    A = Matrix{Float64}(A)
    d = length(mu)

    c = c === nothing ? ones(d) : asvec(c)

    @assert size(A) == (d, d) "A must be d × d."
    @assert length(c) == d "c must have length d."
    @assert p >= 1 "This theorem assumes p ≥ 1."
    @assert all(c .> 0) "All c_i must be positive."

    return RadialTarget(mu, A, Float64(p), c, psi, sample_R, name)
end

dim(t::RadialTarget) = length(t.mu)

function S_value(t::RadialTarget, q::AbstractVector)
    z = t.A * (asvec(q) .- t.mu)
    return sum(t.c .* abs.(z) .^ t.p)
end

energy(t::RadialTarget, q::AbstractVector) = t.psi(S_value(t, q))

function sample_given_R(t::RadialTarget, R::Real; rng::AbstractRNG=Random.default_rng())
    d = dim(t)

    # w ~ Dirichlet(1/p, ..., 1/p)
    w = rand(rng, Dirichlet(fill(1.0 / t.p, d)))

    # independent Rademacher signs
    signs = ifelse.(rand(rng, d) .< 0.5, -1.0, 1.0)

    # y_i = ε_i * (R w_i / c_i)^(1/p)
    y = signs .* ((R .* w ./ t.c) .^ (1.0 / t.p))

    # q = μ + A^{-1} y
    return t.mu .+ (t.A \ y)
end

function sample_target(t::RadialTarget, N::Integer; rng::AbstractRNG=MersenneTwister(0))
    d = dim(t)
    X = Matrix{Float64}(undef, N, d)

    for n in 1:N
        R = t.sample_R(rng)
        q = sample_given_R(t, R; rng=rng)
        @views X[n, :] .= q
    end

    return X
end

function sample_R_values(t::RadialTarget, N::Integer; rng::AbstractRNG=MersenneTwister(0))
    return [t.sample_R(rng) for _ in 1:N]
end

function sample_energy_values(t::RadialTarget, N::Integer; rng::AbstractRNG=MersenneTwister(0))
    return [t.psi(t.sample_R(rng)) for _ in 1:N]
end

function GaussianTarget(; mu, Sigma)
    mu = asvec(mu)
    d = length(mu)
    A = invsqrt_from_scatter(Sigma)

    psi = r -> 0.5 * r
    sample_R = rng -> rand(rng, Gamma(d / 2, 2.0))

    return make_target(
        mu=mu,
        A=A,
        p=2.0,
        c=ones(d),
        psi=psi,
        sample_R=sample_R,
        name="Gaussian"
    )
end

function StudentTTarget(; nu::Real, mu, Sigma)
    @assert nu > 0 "nu must be positive."

    nu = Float64(nu)
    mu = asvec(mu)
    d = length(mu)
    A = invsqrt_from_scatter(Sigma)

    psi = r -> ((nu + d) / 2) * log1p(r / nu)
    sample_R = rng -> nu * rand_betaprime(rng, d / 2, nu / 2)

    return make_target(
        mu=mu,
        A=A,
        p=2.0,
        c=ones(d),
        psi=psi,
        sample_R=sample_R,
        name="Student-t"
    )
end

function CauchyTarget(; mu, Sigma)
    return StudentTTarget(nu=1.0, mu=mu, Sigma=Sigma)
end

function PearsonVIITarget(; kappa::Real, m::Real, mu, Sigma)
    @assert m > 0 "m must be positive."

    kappa = Float64(kappa)
    m = Float64(m)
    mu = asvec(mu)
    d = length(mu)

    @assert kappa > d / 2 "Pearson VII requires kappa > d/2."

    A = invsqrt_from_scatter(Sigma)

    psi = r -> kappa * log1p(r / m)
    sample_R = rng -> m * rand_betaprime(rng, d / 2, kappa - d / 2)

    return make_target(
        mu=mu,
        A=A,
        p=2.0,
        c=ones(d),
        psi=psi,
        sample_R=sample_R,
        name="Pearson VII"
    )
end

function KotzPowerTarget(; beta::Real=1.0, s::Real=1.0, mu, Sigma)
    @assert beta > 0 "beta must be positive."
    @assert s > 0 "s must be positive."

    beta = Float64(beta)
    s = Float64(s)
    mu = asvec(mu)
    d = length(mu)
    A = invsqrt_from_scatter(Sigma)

    psi = r -> beta * r^s
    sample_R = rng -> (rand(rng, Gamma((d / 2) / s, 1.0)) / beta)^(1.0 / s)

    return make_target(
        mu=mu,
        A=A,
        p=2.0,
        c=ones(d),
        psi=psi,
        sample_R=sample_R,
        name="Kotz power-exponential"
    )
end

function FullKotzTarget(; beta::Real=1.0, s::Real=1.0, N0::Real=1.0, mu, Sigma)
    @assert beta > 0 "beta must be positive."
    @assert s > 0 "s must be positive."

    beta = Float64(beta)
    s = Float64(s)
    N0 = Float64(N0)
    mu = asvec(mu)
    d = length(mu)

    @assert d / 2 + N0 - 1 > 0 "Need d/2 + N0 - 1 > 0 for integrability near zero."

    A = invsqrt_from_scatter(Sigma)

    function psi(r)
        if r <= 0
            return N0 == 1.0 ? 0.0 : Inf
        else
            return beta * r^s - (N0 - 1.0) * log(r)
        end
    end

    sample_R = rng -> (rand(rng, Gamma((d / 2 + N0 - 1) / s, 1.0)) / beta)^(1.0 / s)

    return make_target(
        mu=mu,
        A=A,
        p=2.0,
        c=ones(d),
        psi=psi,
        sample_R=sample_R,
        name="Full Kotz"
    )
end

function GeneralPowerTarget(; a::Real=2.0, b::Real=1.0, mu, A, p::Real=2.0, c=nothing)
    @assert a > 0 "a must be positive."
    @assert b > 0 "b must be positive."
    @assert p >= 1 "p must be at least 1."

    a = Float64(a)
    b = Float64(b)
    p = Float64(p)

    mu = asvec(mu)
    d = length(mu)
    A = Matrix{Float64}(A)
    c = c === nothing ? ones(d) : asvec(c)

    alpha = d / p
    eta = a / p

    psi = r -> r^eta / b
    sample_R = rng -> (b * rand(rng, Gamma(alpha / eta, 1.0)))^(1.0 / eta)

    return make_target(
        mu=mu,
        A=A,
        p=p,
        c=c,
        psi=psi,
        sample_R=sample_R,
        name="General power radial"
    )
end

struct TabulatedDensity <: ContinuousUnivariateDistribution
    xs::Vector{Float64}
    fs::Vector{Float64}
end

function Distributions.pdf(d::TabulatedDensity, x::Real)
    xs, fs = d.xs, d.fs

    if x < xs[1] || x > xs[end]
        return 0.0
    end

    i = searchsortedlast(xs, x)

    if i == length(xs)
        return fs[end]
    end

    xL, xR = xs[i], xs[i + 1]
    fL, fR = fs[i], fs[i + 1]

    t = (x - xL) / (xR - xL + eps())
    return (1 - t) * fL + t * fR
end

function normalise_tabulated(xs::Vector{Float64}, f::Vector{Float64})
    area = sum((f[i] + f[i + 1]) * (xs[i + 1] - xs[i]) for i in 1:length(xs)-1) / 2
    fs = area > 0 ? f ./ area : f
    return TabulatedDensity(xs, fs)
end

function tabulated_marginals_2d(t::RadialTarget, X::AbstractMatrix;
        grid_n::Int=800,
        pad_frac::Real=0.5,
        rtol::Real=1e-8,
        atol::Real=1e-10,
        maxevals::Int=1_000_000
    )

    @assert dim(t) == 2 "Numerical marginal plotting is only implemented for d=2."

    xx = X[:, 1]
    yy = X[:, 2]

    xlo, xhi = extrema(xx)
    ylo, yhi = extrema(yy)

    px = pad_frac * (xhi - xlo + eps())
    py = pad_frac * (yhi - ylo + eps())

    xgrid = collect(range(xlo - px, xhi + px, length=grid_n))
    ygrid = collect(range(ylo - py, yhi + py, length=grid_n))

    f1_un(x) = quadgk(
        y -> exp(-energy(t, [x, y])),
        -Inf,
        Inf;
        rtol=rtol,
        atol=atol,
        maxevals=maxevals
    )[1]

    f2_un(y) = quadgk(
        x -> exp(-energy(t, [x, y])),
        -Inf,
        Inf;
        rtol=rtol,
        atol=atol,
        maxevals=maxevals
    )[1]

    fx = map(f1_un, xgrid)
    fy = map(f2_un, ygrid)

    return normalise_tabulated(xgrid, fx), normalise_tabulated(ygrid, fy)
end

function scatter_hist_kde(X::AbstractMatrix;
        target::Union{Nothing,RadialTarget}=nothing,
        bins=:fd,
        bw=nothing,
        top_frac::Real=0.28,
        right_frac::Real=0.34,
        plot_size=(1100, 900),
        ms=2,
        alpha=0.35,
        show_theory::Bool=true,
        theory_style=(linestyle=:dash, color=:black, lw=2.5, alpha=1.0)
    )

    @assert size(X, 2) == 2 "scatter_hist_kde is only for 2D samples."

    xx = X[:, 1]
    yy = X[:, 2]

    xlo, xhi = extrema(xx)
    ylo, yhi = extrema(yy)

    px = 0.05 * (xhi - xlo + eps())
    py = 0.05 * (yhi - ylo + eps())

    xlims = (xlo - px, xhi + px)
    ylims = (ylo - py, yhi + py)

    bwx, bwy = bw === nothing ? (nothing, nothing) :
               bw isa Tuple ? bw :
               (bw, bw)

    p_scatter = scatter(
        xx,
        yy;
        xlims=xlims,
        ylims=ylims,
        xlabel=L"$q_1$",
        ylabel=L"$q_2$",
        legend=false,
        color=RGB(0xdc / 255, 0x39 / 255, 0x12 / 255),
        ms=ms,
        alpha=alpha,
        markerstrokewidth=0,
        aspect_ratio=:auto,
        framestyle=:box,
        grid=true,
        left_margin=8mm,
        right_margin=6mm,
        guidefont=font(20),
        tickfont=font(16)
    )

    p_top = histogram(
        xx;
        bins=bins,
        normalize=:pdf,
        xlims=xlims,
        legend=false,
        xticks=false,
        yaxis=false,
        color=:gray85,
        linecolor=:gray70,
        alpha=0.35,
        framestyle=:none,
        left_margin=8mm,
        right_margin=6mm
    )

    kx = bwx === nothing ? kde(xx) : kde(xx; bandwidth=bwx)

    plot!(
        p_top,
        kx.x,
        kx.density;
        lw=4,
        ls=:solid,
        alpha=0.6,
        color=RGB(0xdc / 255, 0x39 / 255, 0x12 / 255),
        legend=false
    )

    p_right = histogram(
        yy;
        bins=bins,
        normalize=:pdf,
        orientation=:horizontal,
        ylims=ylims,
        legend=false,
        yticks=false,
        xaxis=false,
        color=:gray85,
        linecolor=:gray70,
        alpha=0.35,
        framestyle=:none,
        margin=0mm
    )

    ky = bwy === nothing ? kde(yy) : kde(yy; bandwidth=bwy)

    plot!(
        p_right,
        ky.density,
        ky.x;
        lw=4,
        ls=:solid,
        alpha=0.6,
        color=RGB(0xdc / 255, 0x39 / 255, 0x12 / 255),
        legend=false
    )

    dens_candidates = [maximum(kx.density), maximum(ky.density)]

    if show_theory && target !== nothing
        Dx, Dy = tabulated_marginals_2d(target, X)

        nx = pdf.(Ref(Dx), kx.x)
        ny = pdf.(Ref(Dy), ky.x)

        plot!(p_top, kx.x, nx; legend=false, theory_style...)
        plot!(p_right, ny, ky.x; legend=false, theory_style...)

        push!(dens_candidates, maximum(nx), maximum(ny))
    end

    dens_max = maximum(dens_candidates)

    ylims!(p_top, (0.0, 1.05 * dens_max))
    xlims!(p_right, (0.0, 1.05 * dens_max))

    #blank = plot(; framestyle=:none, axis=false)
    function clean_legend_panel()
        p = plot(
            xlims=(0, 1),
            ylims=(0, 1),
            framestyle=:none,
            axis=false,
            grid=false,
            legend=false,
            margin=0mm
        )

        # Sample dots
        scatter!(p, [0.12-0.06], [0.78];
            ms=5,
            color=RGB(0xdc / 255, 0x39 / 255, 0x12 / 255),
            alpha=0.5,
            markerstrokewidth=0
        )
        annotate!(p, 0.22-0.03, 0.78, text("Sample draws", 14, :left))

        # Histogram bars
        plot!(p, [0.07-0.06, 0.17-0.07], [0.58, 0.58];
            lw=8,
            color=:gray70,
            alpha=0.35
        )
        annotate!(p, 0.22-0.03, 0.58, text("Sample marginals (hist.)", 14, :left))

        # KDEs
        # plot!(p, [0.07, 0.17], [0.38, 0.38];
        #     lw=3,
        #     color=:steelblue,
        #     alpha=0.8
        # )
        plot!(p, [0.07-0.06, 0.17-0.06], [0.35, 0.35];
            lw=3,
            color=RGB(0xdc / 255, 0x39 / 255, 0x12 / 255),
            alpha=0.8
        )
        annotate!(p, 0.22-0.03, 0.345, text("Sample marginals (KDE)", 14, :left))

        # Theory
        plot!(p, [0.07-0.06, 0.11-0.06], [0.13, 0.13];
            lw=2.5,
            color=:black,
            linestyle=:dash
        )
        plot!(p, [0.13-0.06, 0.17-0.06], [0.13, 0.13];
            lw=2.5,
            color=:black,
            linestyle=:dash
        )
        annotate!(p, 0.22-0.03, 0.13, text("Exact marginals", 14, :left))

        return p
    end
    lay = grid(
        2,
        2;
        heights=[top_frac, 1 - top_frac],
        widths=[1 - right_frac, right_frac]
    )
    p_legend = clean_legend_panel()

    return plot(
        p_top,
        p_legend,
        p_scatter,
        p_right;
        layout=lay,
        size=plot_size,
        margin=5mm
    )
end

## Usage

N = 50_000
rng = MersenneTwister(42)

mu = [-1.0, 1.0]

Sigma = [
    1.0  0.6
    0.6  1.5
]

# 1. Gaussian
target = GaussianTarget(mu=mu, Sigma=Sigma)

# 2. Cauchy
#target = CauchyTarget(mu=mu, Sigma=Sigma)

# 3. Student-t
# target = StudentTTarget(nu=5.0, mu=mu, Sigma=Sigma)

# 4. Pearson VII
# target = PearsonVIITarget(kappa=4.0, m=2.0, mu=mu, Sigma=Sigma)

# 5. Kotz / power-exponential
# target = KotzPowerTarget(beta=0.5, s=1.0, mu=mu, Sigma=Sigma)

# 6. Full Kotz with prefactor
# target = FullKotzTarget(beta=0.5, s=1.0, N0=2.0, mu=mu, Sigma=Sigma)
# target = FullKotzTarget(beta=0.1, s=0.5, N0=10.0, mu=mu, Sigma=Sigma)


X = sample_target(target, N; rng=rng)

p = scatter_hist_kde(
    X;
    target=target,
    bins=:fd,
    bw=nothing,
    ms=2,
    alpha=0.35,
    show_theory=true
)

display(p)
