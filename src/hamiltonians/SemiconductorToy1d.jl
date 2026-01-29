
export SemiconductorToy1d
"""
    SemiconductorToy1d{T<:Real} <: GeneralTwoBand{T}

A toy model for a 1D semiconductor with a direct bandgap at the Brillouin zone center.

The Hamiltonian reads 
```math
\\hat{H} = \\frac{\\Delta}{2}\\sigma_x - t \\sin(a k_x/2)\\sigma_z
``` 
with kx=0 the BZ center. The default values when given a [`UnitScaling`](@ref UnitScaling)
object (see Examples below) fit bandgap and dipole at ``k_x=0`` with ``\\hat{x}∥\\Gamma -M``
to the wurtzite ZnO model used in <https://doi.org/10.1103/PhysRevLett.113.073901>.


# Examples
```jldoctest
julia> h = SemiconductorToy1d(UnitScaling(u"1.0fs",u"1.0Å"))
SemiconductorToy1d:
  Δ: 2.507
  t: 6.031
  a: 2.82

```


```jldoctest
julia> h = SemiconductorToy1d(UnitScaling(u"1.0fs",u"1.0Å"))
SemiconductorToy1d:
  Δ: 2.507
  t: 6.031
  a: 2.82

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`GappedDirac`](@ref GappedDirac)
[`MonolayerhBN`](@ref MonolayerhBN) [`BilayerToy`](@ref BilayerToy) 
"""
struct SemiconductorToy1d{T<:Real} <: GeneralTwoBand{T} 
    Δ::T
    t::T
    a::T
end

isperiodic(::SemiconductorToy1d) = true

function SemiconductorToy1d(
    us::UnitScaling,
    gap::Unitful.Energy=u"1.65eV",
    hopping::Unitful.Energy=u"4.30eV",
    latticeconst::Unitful.Length=u"2.82Å")

    return SemiconductorToy1d(
        energyscaled(gap,us),
        energyscaled(hopping,us),
        lengthscaled(latticeconst,us))
end

hx(h::SemiconductorToy1d,kx,ky)    = h.Δ
hx(h::SemiconductorToy1d)          = quote $(h.Δ) end

hy(h::SemiconductorToy1d,kx,ky)    = zero(h.Δ)
hy(h::SemiconductorToy1d)          = quote zero($(h.Δ)) end

hz(h::SemiconductorToy1d,kx,ky)    = -h.t * sin(kx * h.a/2)
hz(h::SemiconductorToy1d)          = quote -$(h.t) * sin(kx * $(h.a)/2) end

dhdkx(h::SemiconductorToy1d,kx,ky) = SA[
    zero(h.Δ),
    zero(h.Δ),
    -h.t * (h.a/2) * cos(kx * h.a/2)]
dhdkx(h::SemiconductorToy1d)       = SA[
    zero(h.Δ),
    zero(h.Δ),
    :(-$(h.t) * ($(h.a)/2) * cos(kx * $(h.a)/2))]

dhdky(h::SemiconductorToy1d,kx,ky) = SA[
    zero(h.Δ),
    zero(h.Δ),
    zero(h.Δ)]
dhdky(h::SemiconductorToy1d)       = SA[
    zero(h.Δ),
    zero(h.Δ),
    zero(h.Δ)]

# Jacobian ∂h_i/∂k_j
jac(h::SemiconductorToy1d,kx,ky) = SA[
    zero(h.Δ)                                   zero(h.Δ)
    zero(h.Δ)                                   zero(h.Δ)
    -h.t * (h.a/2) * cos(kx * h.a/2)             zero(h.Δ)]    

jac(h::SemiconductorToy1d) = @SMatrix [
    quote zero($(h.Δ)) end                          quote zero($(h.Δ)) end
    quote zero($(h.Δ)) end                          quote zero($(h.Δ)) end
    :(-$(h.t) * ($(h.a)/2) * cos(kx * $(h.a)/2))     quote zero($(h.Δ)) end]

function printparamsSI(h::SemiconductorToy1d,us::UnitScaling;digits=3)

    Δ       = energySI(h.Δ,us)
    a       = lengthSI(h.a,us)
    t       = energySI(h.t,us)

    return """
        Δ  = $(round(typeof(Δ),Δ,sigdigits=digits)) ($(h.Δ))
        a  = $(round(typeof(a),a,sigdigits=digits)) ($(h.a))
        t  = $(round(typeof(t),t,sigdigits=digits)) ($(h.t))"""
end

getshortname(h::SemiconductorToy1d)    = "SemiconductorToy1d"
getparams(h::SemiconductorToy1d)       = (Δ=h.Δ,t=h.t,a=h.a)

function getparamsSI(h::SemiconductorToy1d,us::UnitScaling)
    Δ       = energySI(h.Δ,us)
    a       = lengthSI(h.a,us)
    t       = energySI(h.t,us)
    return (Δ=Δ,t=t,a=a)
end

gethvec(h::SemiconductorToy1d) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[Δ,0.0,-t * sin(kx * a/2)]
end

getdhdkx(h::SemiconductorToy1d) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[0.0, 0.0, -t * (a/2) * cos(kx * a/2)]
end

getdhdky(h::SemiconductorToy1d) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[0.0, 0.0, 0.0]
end

# Jacobian ∂h_i/∂k_j
getjac(h::SemiconductorToy1d) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[
        0.0                                   0.0
        0.0                                   0.0
        -t * (a/2) * cos(kx * a/2)            0.0]
end

