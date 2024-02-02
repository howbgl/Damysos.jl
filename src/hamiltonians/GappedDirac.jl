
export GappedDirac
"""
    GappedDirac{T<:Real} <: GeneralTwoBand{T}

Holds the parameter of a dimensionless massive Dirac Hamiltonian.

The Hamiltonian reads 
```math
\\hat{H} = k_x\\sigma_x + k_y\\sigma_y + m\\sigma_z
``` 
such that ``\\vec{h}=[k_x,k_y,m]``.

# Examples
```jldoctest
julia> h = GappedDirac(1.0)
GappedDirac{Float64}(1.0)
```

# See also
[`GeneralTwoBand`](@ref TwoBandHamiltonians.GeneralTwoBand)
"""
struct GappedDirac{T<:Real} <: GeneralTwoBand{T} 
    m::T
end

hx(h::GappedDirac,kx,ky)    = kx
hx(h::GappedDirac)          = quote kx end

hy(h::GappedDirac,kx,ky)    = ky
hy(h::GappedDirac)          = quote ky end

hz(h::GappedDirac,kx,ky)    = h.m
hz(h::GappedDirac)          = quote $(h.m) end

dhdkx(h::GappedDirac,kx,ky) = SA[one(h.m),zero(h.m),zero(h.m)]
dhdkx(h::GappedDirac)       = SA[one(h.m),zero(h.m),zero(h.m)]

dhdky(h::GappedDirac,kx,ky) = SA[zero(h.m),one(h.m),zero(h.m)]
dhdky(h::GappedDirac)       = SA[zero(h.m),one(h.m),zero(h.m)]

# Jacobian ∂h_i/∂k_j
jac(h::GappedDirac,kx,ky) = SA[
    one(h.m) zero(h.m)
    zero(h.m) one(h.m)
    zero(h.m) zero(h.m)]
jac(h::GappedDirac) = SA[
    one(h.m) zero(h.m)
    zero(h.m) one(h.m)
    zero(h.m) zero(h.m)]


function printparamsSI(h::GappedDirac,us::UnitScaling;digits=3)

    m   = energySI(h.m,us)
    vF  = velocitySI(one(h.m),us)
    
    return """
        m  = $(round(typeof(m),m,sigdigits=digits)) ($(h.m))
        vF = $(round(typeof(vF),vF,sigdigits=digits)) ($(1.0))"""
end

getshortname(h::GappedDirac)    = "GappedDirac"
getparams(h::GappedDirac)       = (m=h.m,vF=one(h.m))

function getparamsSI(h::GappedDirac,us::UnitScaling)
    m = energySI(h.m,us)
    vF = velocitySI(one(h.m),us)
    return (m=m,vF=vF)
end

export gethvec
gethvec(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[kx,ky,m]
end

export getdhdx,getdhdy
getdhdx(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[one(m),zero(m),zero(m)]
end
getdhdy(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[zero(m),one(m),zero(m)]
end

export getjac
getjac(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[
        one(h.m) zero(h.m)
        zero(h.m) one(h.m)
        zero(h.m) zero(h.m)]
end

