using Random
import AdvancedHMC

# Refreshes HMC’s momentum without unnecessarily recalculating the target density and gradient.
struct CachedPartialMomentumRefreshment{F<:AbstractFloat} <: AdvancedHMC.AbstractMomentumRefreshment
    α::F
end

# The CMC and tuning kernels use unit-Euclidean Gaussian kinetics.
function AdvancedHMC.refresh(rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
        ref::CachedPartialMomentumRefreshment,
        h::AdvancedHMC.Hamiltonian{<:AdvancedHMC.UnitEuclideanMetric,AdvancedHMC.GaussianKinetic},
        z::AdvancedHMC.PhasePoint)
    r = ref.α * z.r + sqrt(1 - ref.α^2) * AdvancedHMC.rand_momentum(rng,h.metric,h.kinetic,z.θ)
    return AdvancedHMC.phasepoint(h,z.θ,r;ℓπ=z.ℓπ)
end
