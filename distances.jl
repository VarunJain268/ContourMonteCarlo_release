# Calculate KS, CDF-ISE, and histogram KL distances between sample distributions.

using Random

function sorted_finite_float64(x::AbstractVector{<:Real})
    return sort(Float64.(finite_vec(x)))
end

function build_reference_cache(Xeval::AbstractMatrix)
    D, _ = size(Xeval)
    D > 0 || error("Reference matrix must have at least one row.")

    sorted = Vector{Vector{Float64}}(undef, D)
    for d in 1:D
        yd = sorted_finite_float64(vec(@view Xeval[d, :]))
        isempty(yd) && error("Reference dimension $d has no finite samples.")
        sorted[d] = yd
    end

    return (n_dims=D, sorted=sorted)
end

function cdf_metrics_sorted(xs_sorted::AbstractVector{<:Real}, ys_sorted::AbstractVector{<:Real})
    n = length(xs_sorted)
    m = length(ys_sorted)
    (n == 0 || m == 0) && error("Empty input to CDF metrics")

    inv_n = 1.0 / n
    inv_m = 1.0 / m
    i = 1
    j = 1
    seen_x = 0
    seen_y = 0
    ks = 0.0
    cdf_l2 = 0.0
    prev_v = 0.0
    have_prev = false

    while i <= n || j <= m
        xv = i <= n ? Float64(xs_sorted[i]) : Inf
        yv = j <= m ? Float64(ys_sorted[j]) : Inf
        v = xv <= yv ? xv : yv

        if have_prev
            Δ = v - prev_v
            diff = seen_x * inv_n - seen_y * inv_m
            cdf_l2 += diff * diff * Δ
        else
            have_prev = true
        end

        while i <= n && Float64(xs_sorted[i]) == v
            seen_x += 1
            i += 1
        end
        while j <= m && Float64(ys_sorted[j]) == v
            seen_y += 1
            j += 1
        end

        ks = max(ks, abs(seen_x * inv_n - seen_y * inv_m))
        prev_v = v
    end

    return (ks=ks, cdf_l2=cdf_l2)
end

function histogram_edges(x::AbstractVector{<:Real}; nbins::Int)
    nbins > 0 || throw(ArgumentError("nbins must be positive."))
    xs = finite_vec(x)
    isempty(xs) && error("Cannot build histogram edges from empty input.")
    lo, hi = minimum(xs), maximum(xs)
    if lo == hi
        pad = max(abs(lo), 1.0) * 1e-6
        lo -= pad
        hi += pad
    end
    return collect(range(lo, hi; length=nbins + 1))
end

function histogram_probabilities(x::AbstractVector{<:Real}, edges::AbstractVector{<:Real}; pseudocount::Real)
    nbins = length(edges) - 1
    nbins > 0 || throw(ArgumentError("Need at least two histogram edges."))
    counts = fill(Float64(pseudocount), nbins)
    for v_raw in finite_vec(x)
        v = Float64(v_raw)
        b = v >= edges[end] ? nbins : searchsortedlast(edges, v)
        counts[clamp(b, 1, nbins)] += 1.0
    end
    total = sum(counts)
    total > 0 || error("Histogram has no mass.")
    return counts ./ total
end

function binned_kl_to_reference(x::AbstractVector{<:Real}, ref_probs::AbstractVector{<:Real}, edges::AbstractVector{<:Real}; pseudocount::Real)
    p = histogram_probabilities(x, edges; pseudocount=pseudocount)
    length(p) == length(ref_probs) || error("Reference histogram has $(length(ref_probs)) bins, expected $(length(p)).")
    return sum(p .* log.(p ./ ref_probs))
end

per_param_metrics_against_cache(X::AbstractMatrix, ref_cache) =
    per_column_metrics_against_cache(transpose(X), ref_cache)

function per_column_metrics_against_cache(X::AbstractMatrix, ref_cache)
    _, D1 = size(X)
    D1 == ref_cache.n_dims || error("Dimension mismatch: X has D=$D1 but cache has D=$(ref_cache.n_dims)")

    ks = Vector{Float64}(undef, D1)
    l2 = Vector{Float64}(undef, D1)
    for d in 1:D1
        xs = sorted_finite_float64(@view X[:, d])
        isempty(xs) && error("Sample dimension $d has no finite values.")
        out = cdf_metrics_sorted(xs, ref_cache.sorted[d])
        ks[d] = out.ks
        l2[d] = out.cdf_l2
    end
    return (ks=ks, l2=l2)
end
