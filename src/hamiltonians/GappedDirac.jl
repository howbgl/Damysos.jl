
export GappedDirac
"""
    GappedDirac{T<:Real} <: GeneralTwoBand{T}

Massive Dirac Hamiltonian (two-band model).

The Hamiltonian reads 
```math
\\hat{H} = k_x\\sigma_x + k_y\\sigma_y + m\\sigma_z
``` 
such that ``\\vec{h}=[k_x,k_y,m]``.

# Examples
```jldoctest
julia> h = GappedDirac(1.0)
GappedDirac:
  m: 1.0
  vF: 1.0

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`QuadraticToy`](@ref QuadraticToy)
"""
struct GappedDirac{T<:Real} <: GeneralTwoBand{T} 
    m::T
end

function GappedDirac(us::UnitScaling,m::Unitful.Energy,vf::Unitful.Velocity)
    
    delta = uconvert(Unitful.NoUnits,m*timescaleSI(us)/ħ)
    if velocityscaled(vf,us) ≈ 1.0
        return GappedDirac(delta)
    else
        throw(ArgumentError("Scaled velocity must be equal to 1.0"))
    end
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

gethvec(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[kx,ky,m]
end

getdhdkx(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[one(m),zero(m),zero(m)]
end
getdhdky(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[zero(m),one(m),zero(m)]
end

getjac(h::GappedDirac) = let m=h.m
    (kx,ky) -> SA[
        one(m) zero(m)
        zero(m) one(m)
        zero(m) zero(m)]
end

function Base.show(io::IO, ::MIME"text/plain", h::GappedDirac)
	println(io, getshortname(h) * ":")
	print(io, printfields_generic(h) |> prepend_spaces)
    print(io, " vF: 1.0")
end