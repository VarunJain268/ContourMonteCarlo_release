# Use MBAR to combine biased umbrella samples into an estimate of the unbiased energy distribution.

using Optim

# Compute the MBAR denominator term for a single energy value u:
#
#     ∑_{k=1}^K N_k exp(f_k - u - b_k(u)),
#
# in log-space using the log-sum-exp trick for numerical stability.
@inline function logsumexp_k!(work::Vector{Float64}, u::Float64,
    logN_k::Vector{Float64}, f::Vector{Float64},
    centres::Vector{Float64}, inv2σ2::Float64)

    K = length(centres)
    maxlog = -Inf
    @inbounds for j in 1:K
        lj = logN_k[j] + f[j] - u - umbrella_bias(u, centres[j], inv2σ2)
        work[j] = lj
        maxlog = max(maxlog, lj)
    end

    s = 0.0
    @inbounds for j in 1:K
        tj = exp(work[j] - maxlog)
        work[j] = tj
        s += tj
    end

    return maxlog + log(s), s
end

# Compute the MBAR denominator term for all pooled energy samples {u_n}_n=1^N, in log space.
function compute_logdenom(u_all::Vector{Float64},
    logN_k::Vector{Float64}, f::Vector{Float64},
    centres::Vector{Float64}, inv2σ2::Float64)

    K = length(centres)
    N = length(u_all)
    logdenom = Vector{Float64}(undef, N)
    work = Vector{Float64}(undef, K)

    @inbounds for n in 1:N
        logd, _ = logsumexp_k!(work, u_all[n], logN_k, f, centres, inv2σ2)
        logdenom[n] = logd
    end

    return logdenom
end

# Solve the MBAR self-consistency equations using L-BFGS optimisation. Free energies are relative to f[1], fixed at zero; return the pooled energies and log denominators needed for weights.
function mbar_free_energies(energies_by_state::Vector{Vector{Float64}},
    centres::Vector{Float64}, σu::Float64;
    tol::Float64,
    maxiters::Int,
    m::Int,
    show_trace::Bool=false)

    K = length(centres)
    K == length(energies_by_state) || error("centres and energies_by_state mismatch")
    inv2σ2 = 0.5 / (σu^2)

    N_k = [length(energies_by_state[k]) for k in 1:K]
    all(n -> n > 0, N_k) || error("Every umbrella window must contribute at least one energy sample.")
    N = sum(N_k)
    N > 0 || error("No samples provided to MBAR")

    u_all = Vector{Float64}(undef, N)
    idx = 1
    for k in 1:K
        us = energies_by_state[k]
        @inbounds for t in 1:length(us)
            u_all[idx] = us[t]
            idx += 1
        end
    end

    if K == 1
        f = zeros(Float64, 1)
        logN_k = [log(Float64(N_k[1]))]
        logdenom = compute_logdenom(u_all, logN_k, f, centres, inv2σ2)
        return (
            u_all = u_all,
            logdenom = logdenom,
        )
    end

    logN_k = log.(Float64.(N_k))
    f_full = zeros(Float64, K)
    work = Vector{Float64}(undef, K)

    function fg!(F, G, x)
        f_full[1] = 0.0
        @inbounds for j in 2:K
            f_full[j] = x[j - 1]
        end

        if G !== nothing
            fill!(G, 0.0)
        end

        obj = 0.0
        @inbounds for n in 1:N
            u = u_all[n]
            logd, s = logsumexp_k!(work, u, logN_k, f_full, centres, inv2σ2)
            obj += logd

            if G !== nothing
                invs = 1.0 / s
                for j in 2:K
                    G[j - 1] += work[j] * invs
                end
            end
        end

        @inbounds for j in 2:K
            obj -= N_k[j] * f_full[j]
        end

        if G !== nothing
            @inbounds for j in 2:K
                G[j - 1] -= N_k[j]
            end
        end

        return obj
    end

    x0 = zeros(Float64, K - 1)
    res = Optim.optimize(
        Optim.only_fg!(fg!),
        x0,
        Optim.LBFGS(; m=m),
        Optim.Options(
            g_tol=tol,
            iterations=maxiters,
            show_trace=show_trace,
            store_trace=false,
        ),
    )

    xhat = Optim.minimizer(res)
    f = zeros(Float64, K)
    @inbounds for j in 2:K
        f[j] = xhat[j - 1]
    end

    logdenom = compute_logdenom(u_all, logN_k, f, centres, inv2σ2)

    return (
        u_all = u_all,
        logdenom = logdenom,
    )
end

# Compute the MBAR weights \hat W_{0,n} for the unbiased state
# (i.e. the original target distribution) at each pooled energy sample u_n.
function mbar_unbiased_weights(mbar_out)
    logw = @. -mbar_out.u_all - mbar_out.logdenom
    c = maximum(logw)
    w = @. exp(logw - c)
    w ./= sum(w)
    return w
end