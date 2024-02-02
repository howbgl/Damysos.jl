
export getΔϵ
export getϵ
export GeneralTwoBand
export hvec
export Δϵ
export ϵ


"""
    GeneralTwoBand{T} <: Hamiltonian{T}

Supertype of all 2x2 Hamiltonians with all matrixelements via dispatch.

# Idea
The idea is that all Hamiltonians of the form

```math
\\hat{H} = \\vec{h}(\\vec{k})\\cdot\\vec{\\sigma}
```

can be diagonalized analytically and hence most desired matrixelements such as velocities or
dipoles can be expressed solely through 
    
```math
\\vec{h}(\\vec{k})=[h_x(\\vec{k}),h_y(\\vec{k}),h_z(\\vec{k})]
```

and its derivatives with respect to ``k_\\mu``. Any particular Hamiltonian deriving form
GeneralTwoBand{T} must then only implement ``\\vec{h}(\\vec{k})`` and its derivatives.

# See also
[`GappedDirac`](@ref TwoBandHamiltonians.GappedDirac)
"""
abstract type GeneralTwoBand{T} <: Hamiltonian{T} end

"""
    hvec(h,kx,ky)

Returns ``\\vec{h}(\\vec{k})`` defining the Hamiltonian at ``\\vec{k}=[k_x,k_y]``.
"""
hvec(h::GeneralTwoBand,kx,ky) = SA[hx(h,kx,ky),hy(h,kx,ky),hz(h,kx,ky)]
hvec(h::GeneralTwoBand)       = SA[hx(h),hy(h),hz(h)]

"""
    Δϵ(h,kx,ky)

Returns the band energy (valence & conduction) difference at ``\\vec{k}=[k_x,k_y]``.
"""
Δϵ(h::GeneralTwoBand,kx,ky)             = 2ϵ(h,kx,ky)
Δϵ(h::GeneralTwoBand)                   = :(2sqrt($(hx(h))^2+$(hy(h))^2+$(hz(h))^2))
Δϵ(hx::Number,hy::Number,hz::Number)    = 2sqrt(hx^2 + hy^2 + hz^2)
Δϵ(h::SVector{3,<:Number})              = 2sqrt(h[1]^2 + h[2]^2 + h[3]^2)
"""
    ϵ(h,kx,ky)

Returns the eigenenergy of the positive (conduction band) state at ``\\vec{k}=[k_x,k_y]``.
"""
ϵ(h::GeneralTwoBand,kx,ky)  = sqrt(hx(h,kx,ky)^2 + hy(h,kx,ky)^2 + hz(h,kx,ky)^2)
ϵ(hx::Number,hy::Number,hz::Number)     = sqrt(hx^2 + hy^2 + hz^2)
ϵ(h::SVector{3,<:Number})               = sqrt(h[1]^2 + h[2]^2 + h[3]^2)
ϵ(h::GeneralTwoBand)                    = :(sqrt($(hx(h))^2+$(hy(h))^2+$(hz(h))^2))

getΔϵ(hvec::Function)   = (kx,ky) -> 2sqrt(sum(hvec(kx,ky) .^ 2))
getϵ(hvec::Function)    = (kx,ky) -> sqrt(sum(hvec(kx,ky) .^ 2))

include("pauli.jl")
include("velocity.jl")
include("dipole.jl")
