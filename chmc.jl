# Constrained Hamiltonian Monte Carlo on level sets {q : U(q) = u}.

using Random
using LinearAlgebra
using LogDensityProblems
using AdvancedHMC
using Roots

const CHMC_MIN_GRAD_NORM = 1e-12

struct RattleStateCache
    Uq::Float64
    g::Vector{Float64}
end

struct LambdaSolveResult
    ok::Bool
    q1::Vector{Float64}
    p_half::Vector{Float64}
    cache::Union{Nothing,RattleStateCache}
end

struct RattleStepResult
    ok::Bool
    q::Vector{Float64}
    p::Vector{Float64}
    cache::Union{Nothing,RattleStateCache}
end

struct RattleWorkspace
    q_base::Vector{Float64}
    q1::Vector{Float64}
    q_try::Vector{Float64}
    p_half::Vector{Float64}
end

function RattleWorkspace(q::AbstractVector)
    dimension = length(q)
    return RattleWorkspace(
        Vector{Float64}(undef, dimension),
        Vector{Float64}(undef, dimension),
        Vector{Float64}(undef, dimension),
        Vector{Float64}(undef, dimension),
    )
end

@inline allfinite(x) = all(isfinite, x)
@inline potential_energy(ℓπ::LDP, q::AbstractVector{<:Real}) = -LogDensityProblems.logdensity(ℓπ, q)
@inline state_cache(Uq, g) = RattleStateCache(Float64(Uq), Vector{Float64}(g))
@inline fail_lambda(q, p) = LambdaSolveResult(false, Vector{Float64}(q), Vector{Float64}(p), nothing)
@inline ok_lambda(q, p, cache) = LambdaSolveResult(true, Vector{Float64}(q), Vector{Float64}(p), cache)
@inline fail_step(q, p) = RattleStepResult(false, Vector{Float64}(q), Vector{Float64}(p), nothing)

function potential_and_gradient(ℓπ::LDP, q::AbstractVector)
    Uq, g = U_and_∇U(ℓπ, q)
    return isfinite(Uq) && allfinite(g) ? (Float64(Uq), Vector{Float64}(g)) : error("Invalid target state.")
end

function valid_gradient(g::AbstractVector)
    n = norm(g)
    return isfinite(n) && n > CHMC_MIN_GRAD_NORM
end

function U_accept(g::AbstractVector)
    valid_gradient(g) || error("Degenerate gradient in U_accept.")
    return log(norm(g))
end

function H_accept(p::AbstractVector, g::AbstractVector)
    allfinite(p) || error("Non-finite momentum.")
    return 0.5 * dot(p, p) + U_accept(g)
end

function project_momentum_tangent!(p::AbstractVector, g::AbstractVector)
    valid_gradient(g) || error("Degenerate gradient in tangent projection.")
    p .-= (dot(p, g) / dot(g, g)) .* g
    return p
end

function lambda_base!(q_base::Vector{Float64}, q0::Vector{Float64}, p0::Vector{Float64}, h::Real)
    hf = Float64(h)
    @inbounds @simd for k in eachindex(q_base, q0, p0)
        q_base[k] = q0[k] + hf * p0[k]
    end
    return q_base
end

function lambda_position!(q1::Vector{Float64}, q_base::Vector{Float64}, c::Float64, g0::Vector{Float64}, λ::Real)
    cλ = c * Float64(λ)
    @inbounds @simd for k in eachindex(q1, q_base, g0)
        q1[k] = q_base[k] - cλ * g0[k]
    end
    return q1
end

function lambda_half_momentum!(p_half::Vector{Float64}, p0::Vector{Float64}, g0::Vector{Float64}, h::Real, λ::Real)
    h2, λf = Float64(h) / 2, Float64(λ)
    @inbounds @simd for k in eachindex(p_half, p0, g0)
        p_half[k] = p0[k] - h2 * (λf * g0[k])
    end
    return p_half
end

function eval_lambda_residual!(q1::Vector{Float64}, ℓπ::LDP, q_base::Vector{Float64}, g0::Vector{Float64}, u::Real, c::Float64, λ::Real)
    lambda_position!(q1, q_base, c, g0, λ)
    Uq1 = potential_energy(ℓπ, q1)
    return isfinite(Uq1) ? Float64(Uq1 - u) : NaN
end

function finalise_lambda(ℓπ::LDP, q1, p_half)
    try
        Uq1, g1 = potential_and_gradient(ℓπ, q1)
        return ok_lambda(q1, p_half, state_cache(Uq1, g1))
    catch
        return fail_lambda(q1, p_half)
    end
end

function predict_lambda_init(ℓπ::LDP, q_tilde::Vector{Float64}, u::Real, h::Real, g0::Vector{Float64}; predictor_mode::Symbol)
    gg0 = dot(g0, g0)
    h2 = Float64(h)^2
    if predictor_mode === :improved
        U_tilde, g_tilde = U_and_∇U(ℓπ, q_tilde)
        denom = h2 * dot(g_tilde, g0)
    elseif predictor_mode === :cheap
        U_tilde = potential_energy(ℓπ, q_tilde)
        denom = h2 * gg0
    else
        error("Unknown predictor_mode: $predictor_mode. Use :improved or :cheap.")
    end
    return isfinite(U_tilde) && isfinite(denom) && abs(denom) > 1e-30 ?
        Float64(2 * (U_tilde - u) / denom) : -0.0
end

function solve_lambda_secant(ℓπ::LDP, ws::RattleWorkspace, p0::Vector{Float64}, u::Real, h::Real,
                             g0::Vector{Float64}, q_base::Vector{Float64}, c::Float64, λ_init::Real;
                             maxiters::Integer, secant_init_width::Real,
                             root_residual_tol::Real, root_xtol::Real,
                             root_rtol::Real)
    f0 = eval_lambda_residual!(ws.q1, ℓπ, q_base, g0, u, c, λ_init)
    if isfinite(f0) && abs(f0) <= root_residual_tol
        lambda_half_momentum!(ws.p_half, p0, g0, h, λ_init)
        return finalise_lambda(ℓπ, ws.q1, ws.p_half)
    end
    Δ = max(Float64(secant_init_width), abs(Float64(λ_init)))
    λ0, λ1 = Float64(λ_init) - Δ, Float64(λ_init) + Δ
    λ0 == λ1 && (λ1 += Float64(secant_init_width))
    residual = λ -> eval_lambda_residual!(ws.q_try, ℓπ, q_base, g0, u, c, Float64(λ))
    λ_root = try
        Roots.find_zero(residual, (λ0, λ1), Roots.Secant();
            xatol=Float64(root_xtol), xrtol=Float64(root_rtol), maxiters=Int(maxiters))
    catch
        lambda_half_momentum!(ws.p_half, p0, g0, h, λ_init)
        return fail_lambda(ws.q1, ws.p_half)
    end
    f = eval_lambda_residual!(ws.q1, ℓπ, q_base, g0, u, c, λ_root)
    lambda_half_momentum!(ws.p_half, p0, g0, h, λ_root)
    isfinite(f) || return fail_lambda(ws.q1, ws.p_half)
    return finalise_lambda(ℓπ, ws.q1, ws.p_half)
end

function solve_lambda_scalar(
    ℓπ::LDP,
    ws::RattleWorkspace,
    q0::Vector{Float64},
    p0::Vector{Float64},
    u::Real,
    h::Real,
    g0::Vector{Float64};
    predictor_mode::Symbol,
    maxiters::Integer,
    secant_init_width::Real,
    root_residual_tol::Real,
    root_xtol::Real,
    root_rtol::Real,
)
    c = Float64(h)^2 / 2
    q_base = lambda_base!(ws.q_base, q0, p0, h)
    λ_init = predict_lambda_init(ℓπ, q_base, u, h, g0; predictor_mode=predictor_mode)
    return solve_lambda_secant(
        ℓπ, ws, p0, u, h, g0, q_base, c, λ_init;
        maxiters=maxiters,
        secant_init_width=secant_init_width,
        root_residual_tol=root_residual_tol,
        root_xtol=root_xtol,
        root_rtol=root_rtol,
    )
end

# Zero-guidance RATTLE: enforce the energy contour and tangent momentum.
function rattle_step(
    ℓπ::LDP,
    q0::Vector{Float64},
    p0::Vector{Float64},
    u::Real,
    h::Real,
    ws::RattleWorkspace;
    start_cache::RattleStateCache,
    predictor_mode::Symbol,
    constraint_tol::Real,
    root_maxiters::Integer,
    secant_init_width::Real,
    root_residual_tol::Real,
    root_xtol::Real,
    root_rtol::Real,
    constraint_solve_counter::Base.RefValue{Int},
)
    cache0 = start_cache
    valid_gradient(cache0.g) || return fail_step(q0, p0)

    constraint_solve_counter[] += 1
    solve_out = solve_lambda_scalar(
        ℓπ, ws, q0, p0, u, h, cache0.g;
        predictor_mode=predictor_mode,
        maxiters=root_maxiters,
        secant_init_width=secant_init_width,
        root_residual_tol=root_residual_tol,
        root_xtol=root_xtol,
        root_rtol=root_rtol,
    )
    solve_out.ok || return fail_step(q0, p0)
    g1 = solve_out.cache.g
    Uq1 = solve_out.cache.Uq
    valid_gradient(g1) || return fail_step(q0, p0)

    μ = ((2 / h) * dot(g1, solve_out.p_half)) / dot(g1, g1)
    isfinite(μ) || return fail_step(q0, p0)
    p1 = solve_out.p_half .- (h / 2) .* (μ .* g1)
    allfinite(p1) && abs(Uq1 - u) <= constraint_tol || return fail_step(q0, p0)
    return RattleStepResult(true, solve_out.q1, Vector{Float64}(p1), solve_out.cache)
end

function integrate_rattle(ℓπ::LDP, q0::Vector{Float64}, p0::Vector{Float64}, u::Real; h::Real, L::Integer,
                          start_cache::RattleStateCache,
                          workspace::RattleWorkspace, kwargs...)
    L >= 1 || throw(ArgumentError("L must be >= 1"))
    q, p, cache = copy(q0), copy(p0), start_cache
    for _ in 1:L
        out = rattle_step(ℓπ, q, p, u, h, workspace; start_cache=cache, kwargs...)
        out.ok || return fail_step(q0, p0)
        q, p, cache = out.q, out.p, out.cache
    end
    return RattleStepResult(true, q, p, cache)
end

function rattle_roundtrip_matches(q_back, p_back, q0, p0; atol::Real, rtol::Real)
    return all(i -> isfinite(q_back[i]) &&
        abs(q_back[i] - q0[i]) <= atol + rtol * max(abs(q_back[i]), abs(q0[i])), eachindex(q0)) &&
        all(i -> isfinite(p_back[i]) &&
        abs(p_back[i] + p0[i]) <= atol + rtol * max(abs(p_back[i]), abs(p0[i])), eachindex(p0))
end

function validate_reversibility_tolerances(atol, rtol)
    isfinite(atol) && isfinite(rtol) && atol >= 0 && rtol >= 0 && (atol > 0 || rtol > 0) ||
        throw(ArgumentError("Reversibility tolerances must be finite, nonnegative, and not both zero."))
end

function chmc_transition(
    rng::AbstractRNG,
    ℓπ::LDP,
    q0::Vector{Float64},
    cache0::Union{Nothing,RattleStateCache},
    u::Real;
    h::Real,
    L::Integer,
    workspace::RattleWorkspace,
    reverse_workspace::RattleWorkspace,
    reversibility_atol::Real,
    reversibility_rtol::Real,
    reverse_rejection_counter::Base.RefValue{Int},
    kwargs...,
)
    validate_reversibility_tolerances(reversibility_atol, reversibility_rtol)
    cache_start = if cache0 === nothing
        try
            state_cache(potential_and_gradient(ℓπ, q0)...)
        catch
            return q0, nothing, false, 0.0, false
        end
    else
        cache0
    end
    p0 = try
        project_momentum_tangent!(randn(rng, length(q0)), cache_start.g)
    catch
        return q0, cache_start, false, 0.0, false
    end

    traj = integrate_rattle(
        ℓπ, q0, p0, u;
        h = h,
        L = L,
        start_cache = cache_start,
        workspace = workspace,
        kwargs...,
    )
    traj.ok || return q0, cache_start, false, 0.0, false
    H0, HL = try
        H_accept(p0, cache_start.g), H_accept(traj.p, traj.cache.g)
    catch
        return q0, cache_start, false, 0.0, false
    end
    ΔH = H0 - HL
    # Final acceptance requires both MH and reversibility. Short-circuit MH
    # rejections before spending gradients on the reverse trajectory.
    log(rand(rng)) <= min(0.0, ΔH) || return q0, cache_start, false, ΔH, true

    reverse = integrate_rattle(
        ℓπ, traj.q, -traj.p, u;
        h = h,
        L = L,
        start_cache = traj.cache,
        workspace = reverse_workspace,
        kwargs...,
    )
    if !reverse.ok || !rattle_roundtrip_matches(reverse.q, reverse.p, q0, p0;
            atol = reversibility_atol, rtol=reversibility_rtol)
        reverse_rejection_counter[] += 1
        return q0, cache_start, false, 0.0, false
    end
    return traj.q, traj.cache, true, ΔH, true
end

function count_state_changes(X::AbstractMatrix, q0::AbstractVector)
    n = size(X, 2)
    n == 0 && return 0
    acc, prev = 0, q0
    for j in 1:n
        changed = any(k -> X[k, j] != prev[k], 1:size(X, 1))
        acc += changed
        prev = view(X, :, j)
    end
    return acc
end

function aggregate_cumulative_int_traces(traces::AbstractVector{<:AbstractVector{<:Integer}})
    total = zeros(Int, length(traces[1]))
    for trace in traces
        total .+= Int.(trace)
    end
    return (avg_per_chain=Float64.(total) ./ length(traces), total=total)
end

function run_hmc_from_endpoint(
    rng::AbstractRNG,
    ℓπ::LDP,
    q0::Vector{Float64};
    S::Int,
    h::Real,
    L::Int,
    momentum_refresh::Real,
)
    d = length(q0)
    metric = UnitEuclideanMetric(d)
    hamiltonian = Hamiltonian(metric, ℓπ)
    kernel = build_kernel(
        Leapfrog(Float64(h));
        momentum_refresh=momentum_refresh,
        L=L,
    )
    before_initial = read_target_grad_counter()
    z = AdvancedHMC.phasepoint(rng, copy(q0), hamiltonian)
    initial_grad_evals = read_target_grad_counter() - before_initial
    X = Matrix{Float64}(undef, d, S)
    cumulative_grad_evals = zeros(Int, S + 1)
    for j in 1:S
        grads_before = read_target_grad_counter()
        t = AdvancedHMC.transition(rng, hamiltonian, kernel, z)
        z = t.z
        X[:, j] .= z.θ
        cumulative_grad_evals[j + 1] = cumulative_grad_evals[j] +
            (read_target_grad_counter() - grads_before) + (j == 1 ? initial_grad_evals : 0)
    end
    return (
        samples = X,
        acc_rate = count_state_changes(X, q0) / S,
        cumulative_grad_evals = cumulative_grad_evals,
    )
end

function run_chmc(
    rng::AbstractRNG;
    ℓπ::LDP,
    us::Vector{Float64},
    S::Int,
    h::Real,
    L::Int,
    predictor_mode::Symbol,
    alpha::Real,
    h_hmc::Real,
    L_hmc::Int,
    hmc_momentum_refresh::Real = 0.0,
    constraint_tol::Real,
    root_maxiters::Int,
    secant_init_width::Real,
    root_residual_tol::Real,
    root_xtol::Real,
    root_rtol::Real,
    reversibility_atol::Real,
    reversibility_rtol::Real,
    q0s::AbstractMatrix,
    chain_workers::Int,
)
    # Validate settings and split the run into CHMC and HMC steps.
    validate_reversibility_tolerances(reversibility_atol, reversibility_rtol)
    predictor_mode in (:improved, :cheap) || error("predictor_mode must be :improved or :cheap.")
    0 <= alpha <= 1 || throw(ArgumentError("alpha must be in [0, 1]."))
    S >= 0 && L >= 1 && h > 0 && L_hmc >= 1 && h_hmc > 0 ||
        throw(ArgumentError("Invalid iteration or step-size settings."))
    secant_init_width > 0 || throw(ArgumentError("secant_init_width must be positive."))
    root_residual_tol > 0 && root_xtol > 0 && root_rtol >= 0 ||
        throw(ArgumentError("Invalid root tolerances."))

    dimension = Int(LogDensityProblems.dimension(ℓπ))
    n_chains = length(us)
    size(q0s) == (dimension, n_chains) ||
        throw(ArgumentError("q0s must have size ($dimension, $n_chains), got $(size(q0s))."))
    n_chmc = clamp(round(Int, alpha * S), 0, S)
    n_hmc = S - n_chmc

    # Allocate outputs before scheduling chains.
    chains = [Matrix{Float64}(undef, dimension, S + 1) for _ in 1:n_chains]
    cumulative_grad_evals_by_chain = [zeros(Int, S + 1) for _ in 1:n_chains]
    chmc_accept_counts = zeros(Int, n_chains)
    hmc_accept_rate = fill(NaN, n_chains)
    mean_energy_change = fill(NaN, n_chains)
    max_constraint_errors = zeros(Float64, n_chains)
    rattle_failures = zeros(Int, n_chains) # Forward failures + failed reverse checks on MH survivors.
    reverse_rejections = zeros(Int, n_chains)
    constraint_solves = zeros(Int, n_chains) # Scalar solve attempts, including failures and reverse trajectories.

    validate_chain_workers(chain_workers)
    check_chain_target(ℓπ, chain_workers)
    chain_rngs = [MersenneTwister(rand(rng, UInt)) for _ in 1:n_chains]
    target_grad_evals_by_chain = run_chain_jobs(n_chains; chain_workers=chain_workers) do chain_index
        chain_rng = chain_rngs[chain_index]
        u = us[chain_index]
        position = Vector{Float64}(view(q0s, :, chain_index))
        chains[chain_index][:, 1] .= position
        forward_workspace = RattleWorkspace(position)
        reverse_workspace = RattleWorkspace(position)
        constraint_solve_counter = Ref(0)
        reverse_rejection_counter = Ref(0)
        cache = nothing
        sum_energy_change = 0.0
        max_constraint_error = 0.0

        # Run CHMC while staying on this chain's energy contour.
        for step_index in 1:n_chmc
            gradients_before = read_target_grad_counter()
            position, cache, accepted, ΔH, trajectory_ok = chmc_transition(
                chain_rng, ℓπ, position, cache, u;
                h = h,
                L = L,
                predictor_mode = predictor_mode,
                constraint_tol = constraint_tol,
                workspace = forward_workspace,
                reverse_workspace = reverse_workspace,
                reversibility_atol = reversibility_atol,
                reversibility_rtol = reversibility_rtol,
                reverse_rejection_counter = reverse_rejection_counter,
                constraint_solve_counter = constraint_solve_counter,
                root_maxiters = root_maxiters,
                secant_init_width = secant_init_width,
                root_residual_tol = root_residual_tol,
                root_xtol = root_xtol,
                root_rtol = root_rtol,
            )

            chains[chain_index][:, step_index + 1] .= position
            cumulative_grad_evals_by_chain[chain_index][step_index + 1] =
                cumulative_grad_evals_by_chain[chain_index][step_index] +
                (read_target_grad_counter() - gradients_before)
            chmc_accept_counts[chain_index] += accepted
            rattle_failures[chain_index] += !trajectory_ok
            sum_energy_change += ΔH
            potential = cache === nothing ? potential_energy(ℓπ, position) : cache.Uq
            isfinite(potential) ||
                error("Encountered non-finite potential while recording chain $chain_index.")
            max_constraint_error = max(max_constraint_error, abs(potential - u))
        end

        # Continue with ordinary HMC from the final CHMC position.
        if n_hmc > 0
            hmc_out = run_hmc_from_endpoint(
                chain_rng, ℓπ, position;
                S = n_hmc,
                h = h_hmc,
                L = L_hmc,
                momentum_refresh = hmc_momentum_refresh,
            )
            chains[chain_index][:, n_chmc + 2:S + 1] .= hmc_out.samples
            hmc_accept_rate[chain_index] = hmc_out.acc_rate
            gradient_offset = cumulative_grad_evals_by_chain[chain_index][n_chmc + 1]
            for step_index in 1:n_hmc
                cumulative_grad_evals_by_chain[chain_index][n_chmc + 1 + step_index] =
                    gradient_offset + hmc_out.cumulative_grad_evals[step_index + 1]
            end
        end

        constraint_solves[chain_index] = constraint_solve_counter[]
        reverse_rejections[chain_index] = reverse_rejection_counter[]
        mean_energy_change[chain_index] = n_chmc > 0 ? sum_energy_change / n_chmc : NaN
        max_constraint_errors[chain_index] = max_constraint_error
    end

    return (
        us = us,
        chains = chains,
        chain_workers = chain_workers,
        target_grad_evals_by_chain = target_grad_evals_by_chain,
        transition_step = n_chmc,
        n_chmc = n_chmc,
        n_hmc = n_hmc,
        cumulative_grad_evals_by_chain = cumulative_grad_evals_by_chain,
        gradient_eval_xaxes = aggregate_cumulative_int_traces(cumulative_grad_evals_by_chain),
        predictor_mode = predictor_mode,
        alpha = alpha,
        chmc_accept_rate = n_chmc > 0 ? chmc_accept_counts ./ n_chmc : fill(NaN, n_chains),
        hmc_accept_rate = hmc_accept_rate,
        mean_ΔH = mean_energy_change,
        max_constraint_err = max_constraint_errors,
        reversibility_atol = reversibility_atol,
        reversibility_rtol = reversibility_rtol,
        reverse_check_rejections = reverse_rejections,
        constraint_solves_by_chain = constraint_solves,
        rattle_failures = rattle_failures,
        rattle_fail_rate = n_chmc > 0 ? rattle_failures ./ n_chmc : fill(NaN, n_chains),
    )
end
