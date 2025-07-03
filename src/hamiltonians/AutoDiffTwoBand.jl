export AutoDiffTwoBand
"""
    GappedDirac{T<:Real} <: GeneralTwoBand{T}

Arbitrary two-band Hamiltonian with auto-differentiation.

The Hamiltonian reads 
```math
\\hat{H} = h_x\\sigma_x + h_y\\sigma_y + h_z\\sigma_z
``` 
such that ``\\vec{h}=[h_x,h_y,h_z]``.

# Examples
```jldoctest
julia> h = GappedDirac(1.0)
GappedDirac:
  m: 1.0
  vF: 1.0

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`QuadraticToy`](@ref QuadraticToy)
[`BilayerToy`](@ref BilayerToy)
"""
struct AutoDiffTwoBand{T<:Real} <: GeneralTwoBand{T} 
    h::Function
end


hx(h::AutoDiffTwoBand,kx,ky)    = h.h(SA[kx,ky])[1]
hx(h::AutoDiffTwoBand)          = quote $(h.h)(SA[kx,ky])[1] end

hy(h::AutoDiffTwoBand,kx,ky)    = h.h(SA[kx,ky])[2]
hy(h::AutoDiffTwoBand)          = quote $(h.h)(SA[kx,ky])[2] end

hz(h::AutoDiffTwoBand,kx,ky)    = h.h(SA[kx,ky])[3]
hz(h::AutoDiffTwoBand)          = quote $(h.h)(SA[kx,ky])[3] end

dhdkx(h::AutoDiffTwoBand,kx,ky) = ForwardDiff.jacobian(h.h,SA[kx,ky])[:,1]
dhdkx(h::AutoDiffTwoBand)       = :(ForwardDiff.jacobian($(h.h),SA[kx,ky])[:,1])

dhdky(h::AutoDiffTwoBand,kx,ky) = ForwardDiff.jacobian(h.h,SA[kx,ky])[:,2]
dhdky(h::AutoDiffTwoBand)       = :(ForwardDiff.jacobian($(h.h),SA[kx,ky])[:,2])

# Jacobian ∂h_i/∂k_j
jac(h::AutoDiffTwoBand,kx,ky) = ForwardDiff.jacobian(h.h,SA[kx,ky])
jac(h::AutoDiffTwoBand)       = :(ForwardDiff.jacobian($(h.h),SA[kx,ky]))

