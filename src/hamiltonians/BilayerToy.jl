
export BilayerToy
"""
    BilayerToy{T<:Real} <: GeneralTwoBand{T}

A toy model of touching quadratic bands with a gap turning them quartic near the band edge.

The Hamiltonian reads 
```math
\\hat{H} = \\frac{\\zeta}{2}(k_x^2 + k_y^2)\\sigma_x + \\frac{\\Delta}{2}\\sigma_z
``` 
such that ``\\vec{h}=[\\zeta/2 (k_x^2 + k_y^2), 0, \\Delta/2]``. The dimensionful 
form (SI) would be
```math
\\hat{H}_{SI} = \\frac{\\hbar^2}{2m^*}(k_x^2+k_y^2)\\sigma_x+\\frac{E_{gap}}{2}\\sigma_z
```

# Examples
```jldoctest
julia> h = BilayerToy(0.2,1.0)
BilayerToy:
  Δ: 0.2
  ζ: 1.0
 

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`GappedDirac`](@ref GappedDirac)
"""
struct BilayerToy{T<:Real} <: GeneralTwoBand{T} 
    Δ::T
    ζ::T
end

function BilayerToy(us::UnitScaling,gap::Unitful.Energy,mass::Unitful.Mass)
    
    lc = lengthscaleSI(us)
    tc = timescale(us)
    delta = uconvert(Unitful.NoUnits,gap * tc / Unitful.ħ)
    zeta  = uconvert(Unitful.NoUnits,Unitful.ħ * tc / (lc^2 * mass))
    return BilayerToy(delta,zeta)
end

hx(h::BilayerToy,kx,ky)    = h.ζ * kx^2 / 2
hx(h::BilayerToy)          = quote $(h.ζ/2) * kx^2 end

hy(h::BilayerToy,kx,ky)    = h.ζ * ky^2 / 2 
hy(h::BilayerToy)          = quote $(h.ζ/2) * ky^2 end

hz(h::BilayerToy,kx,ky)    = h.Δ / 2
hz(h::BilayerToy)          = quote $(h.Δ / 2) end

dhdkx(h::BilayerToy,kx,ky) = SA[h.ζ * kx,zero(h.Δ),zero(h.Δ)]
dhdkx(h::BilayerToy)       = SA[:($(h.ζ)*kx),zero(h.Δ),zero(h.Δ)]

dhdky(h::BilayerToy,kx,ky) = SA[zero(h.Δ),h.ζ * ky,zero(h.Δ)]
dhdky(h::BilayerToy)       = SA[zero(h.Δ),:($(h.ζ)*ky),zero(h.Δ)]

# Jacobian ∂h_i/∂k_j
jac(h::BilayerToy,kx,ky) = SA[
    h.ζ * kx zero(h.Δ)
    zero(h.Δ) h.ζ * ky
    zero(h.Δ) zero(h.Δ)]
jac(h::BilayerToy) = @SMatrix [
    :($(h.ζ)*kx)            quote zero($(h.Δ)) end
    quote zero($(h.Δ)) end     :($(h.ζ)*ky)
    quote zero($(h.Δ)) end     quote zero($(h.Δ)) end]


function printparamsSI(h::BilayerToy,us::UnitScaling;digits=3)

    tc      = timescaleSI(us)
    lc      = timescaleSI(us)
    Δ       = energySI(h.Δ,us)
    m       = uconvert(u"kg",ħ*tc / (h.Δ*lc^2))
    
    return """
        Δ  = $(round(typeof(Δ),Δ,sigdigits=digits)) ($(h.Δ))
        m* = $(round(typeof(m),m,sigdigits=digits)) ($(1/h.ζ))"""
end

getshortname(h::BilayerToy)    = "BilayerToy"
getparams(h::BilayerToy)       = (Δ=h.Δ,ζ=h.ζ)

function getparamsSI(h::BilayerToy,us::UnitScaling)
    Δ = energySI(h.Δ,us)
    m = massSI(1/h.ζ,us)
    return (m=m,Δ=Δ)
end

gethvec(h::BilayerToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[ζ*kx^2 / 2,ζ *ky^2 / 2,Δ/2]
end

getdhdkx(h::BilayerToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[ζ*kx,zero(Δ),zero(Δ)]
end
getdhdky(h::BilayerToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[zero(Δ),ζ*ky,zero(Δ)]
end

getjac(h::BilayerToy) = let Δ=h.Δ,ζ=h.ζ
    (kx,ky) -> SA[
        ζ*kx zero(Δ)
        zero(Δ) ζ*ky
        zero(Δ) zero(Δ)]
end

