
export getΔϵ
export getϵ
export GeneralTwoBand
export hmat
export hvec
export Δϵ
export ϵ

const BAND_SYMBOLS = (:ϵ,:Δϵ,:getϵ,:getΔϵ,:hvec,:hmat)

σ0(::Type{T}) where T<:Real = SMatrix{2,2,Complex{T}}(one(T),0,0,one(T))
σx(::Type{T}) where T<:Real = SMatrix{2,2,Complex{T}}(0,one(T),one(T),0)
σy(::Type{T}) where T<:Real = SMatrix{2,2,Complex{T}}(0,im,-im,0)
σz(::Type{T}) where T<:Real = SMatrix{2,2,Complex{T}}(one(T),0,0,-one(T))

paulivector(::Type{T}) where T<:Real  = SA[σx(T),σy(T),σz(T)]
paulivector(::Hamiltonian{T}) where T = paulivector(T)
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
[`GappedDirac`](@ref GappedDirac)
"""
abstract type GeneralTwoBand{T} <: Hamiltonian{T} end

"""
    hvec(h,kx,ky)

Returns ``\\vec{h}(\\vec{k})`` defining the Hamiltonian at ``\\vec{k}=[k_x,k_y]``.
"""
hvec(h::GeneralTwoBand,kx,ky) = SA[hx(h,kx,ky),hy(h,kx,ky),hz(h,kx,ky)]
hvec(h::GeneralTwoBand)       = SA[hx(h),hy(h),hz(h)]


"""
    hmat(h,kx,ky)

Returns matrixelements of Hamiltonian `h` at ``\\vec{k}=[k_x,k_y]`` (in diabatic basis).
"""
hmat(h::GeneralTwoBand{T},kx,ky) where T = Hermitian(sum(paulivector(T) .* hvec(h,kx,ky)))


function eigvecs_numeric(h::GeneralTwoBand,kx,ky)
    es = eigen(hmat(h,kx,ky))
    vb = es.vectors[:,1]
    cb = es.vectors[:,2]
    
    # Fix EVs to have the same gauge as the analytical expressions
    vb *= exp(-im*angle(vb[2]))
    cb *= exp(-im*angle(cb[1]))
    return hcat(cb,vb)
end

function adiabatic_melements_numeric(h::GeneralTwoBand,op::AbstractMatrix,kx,ky)
    u = eigvecs_numeric(h,kx,ky)
    return transpose(conj(u)) * op * u
end

"Estimate the largest numeric scale in a Hamiltonian"
function estimate_atol(h::GeneralTwoBand)
    return 1e-10*maximum([getproperty(h,s) for s in fieldnames(typeof(h))])
end

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

getΔϵ(hvec::Function)       = (kx,ky) -> 2sqrt(sum(hvec(kx,ky) .^ 2))
getΔϵ(h::GeneralTwoBand)    = getΔϵ(gethvec(h))
getϵ(hvec::Function)        = (kx,ky) -> sqrt(sum(hvec(kx,ky) .^ 2))
getϵ(h::GeneralTwoBand)     = getϵ(gethvec(h))

include("pauli.jl")
include("velocity.jl")
include("dipole.jl")
