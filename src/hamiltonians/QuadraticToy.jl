
export QuadraticToy
"""
    QuadraticToy{T<:Real} <: GeneralTwoBand{T}

A toy model of quadratic bands with a gap.

The Hamiltonian reads 
```math
\\hat{H} = \\frac{\\zeta}{2}(k_x^2\\sigma_x + k_y^2\\sigma_y) + \\frac{\\Delta}{2}\\sigma_z
``` 
such that ``\\vec{h}=[\\zeta/2 k_x^2,\\zeta/2 k_y^2, \\Delta/2]``. The dimensionful 
form (SI) would be
```math
\\hat{H}_{SI} = \\frac{\\hbar^2}{2m^*}(k_x^2\\sigma_x+k_y^2\\sigma_y)+\\frac{E_{gap}}{2}\\sigma_z
```

# Examples
```jldoctest
julia> h = QuadraticToy(0.2,1.0)
QuadraticToy:
  Δ: 0.2
  ζ: 1.0
 

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`GappedDirac`](@ref GappedDirac)
"""
struct QuadraticToy{T<:Real} <: GeneralTwoBand{T} 
    Δ::T
    ζ::T
end

function QuadraticToy(us::UnitScaling,gap::Unitful.Energy,mass::Unitful.Mass)
    
    lc = lengthscaleSI(us)
    tc = timescale(us)
    delta = uconvert(Unitful.NoUnits,gap * tc / Unitful.ħ)
    zeta  = uconvert(Unitful.NoUnits,Unitful.ħ * tc / (lc^2 * mass))
    return QuadraticToy(delta,zeta)
end

isperiodic(::QuadraticToy) = false

hx(h::QuadraticToy,kx,ky)    = h.ζ * kx^2 / 2
hx(h::QuadraticToy)          = quote $(h.ζ/2) * kx^2 end

hy(h::QuadraticToy,kx,ky)    = h.ζ * ky^2 / 2 
hy(h::QuadraticToy)          = quote $(h.ζ/2) * ky^2 end

hz(h::QuadraticToy,kx,ky)    = h.Δ / 2
hz(h::QuadraticToy)          = quote $(h.Δ / 2) end

dhdkx(h::QuadraticToy,kx,ky) = SA[h.ζ * kx,zero(h.Δ),zero(h.Δ)]
dhdkx(h::QuadraticToy)       = SA[:($(h.ζ)*kx),zero(h.Δ),zero(h.Δ)]

dhdky(h::QuadraticToy,kx,ky) = SA[zero(h.Δ),h.ζ * ky,zero(h.Δ)]
dhdky(h::QuadraticToy)       = SA[zero(h.Δ),:($(h.ζ)*ky),zero(h.Δ)]

# Jacobian ∂h_i/∂k_j
jac(h::QuadraticToy,kx,ky) = SA[
    h.ζ * kx zero(h.Δ)
    zero(h.Δ) h.ζ * ky
    zero(h.Δ) zero(h.Δ)]
jac(h::QuadraticToy) = @SMatrix [
    :($(h.ζ)*kx)            quote zero($(h.Δ)) end
    quote zero($(h.Δ)) end     :($(h.ζ)*ky)
    quote zero($(h.Δ)) end     quote zero($(h.Δ)) end]


function printparamsSI(h::QuadraticToy,us::UnitScaling;digits=3)

    tc      = timescaleSI(us)
    lc      = timescaleSI(us)
    Δ       = energySI(h.Δ,us)
    m       = uconvert(u"kg",ħ*tc / (h.Δ*lc^2))
    
    return """
        Δ  = $(round(typeof(Δ),Δ,sigdigits=digits)) ($(h.Δ))
        m* = $(round(typeof(m),m,sigdigits=digits)) ($(1/h.ζ))"""
end

getshortname(h::QuadraticToy)    = "QuadraticToy"
getparams(h::QuadraticToy)       = (Δ=h.Δ,ζ=h.ζ)

function getparamsSI(h::QuadraticToy,us::UnitScaling)
    Δ = energySI(h.Δ,us)
    m = massSI(1/h.ζ,us)
    return (m=m,Δ=Δ)
end

gethvec(h::QuadraticToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[ζ*kx^2 / 2,ζ *ky^2 / 2,Δ/2]
end

getdhdkx(h::QuadraticToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[ζ*kx,zero(Δ),zero(Δ)]
end
getdhdky(h::QuadraticToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[zero(Δ),ζ*ky,zero(Δ)]
end

getjac(h::QuadraticToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[
        ζ*kx zero(Δ)
        zero(Δ) ζ*ky
        zero(Δ) zero(Δ)]
end

