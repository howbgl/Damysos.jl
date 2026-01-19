
export MonolayerhBN
"""
    MonolayerhBN{T<:Real} <: GeneralTwoBand{T}

A tight-binding model for monolayer hexagonal boron nitride (hBN).

The Hamiltonian reads 
```math
\\hat{H} = t[1+2\\sin(a k_x/2)\\cos(\\sqrt{3} k_y a/2)]\\sigma_x - 2t \\cos(a k_x/2)\\sin(\\sqrt{3} k_y a/2)\\sigma_y + \\frac{\\Delta}{2}\\sigma_z
``` 
see  <https://doi.org/10.1103/PhysRevB.83.235312> for details.


# Examples
```jldoctest
julia> h = MonolayerhBN(0.2,1.0, 1.0)
MonolayerhBN:
  Δ: 0.2
  t: 1.0
  a: 1.0

```

# See also
[`GeneralTwoBand`](@ref GeneralTwoBand) [`GappedDirac`](@ref GappedDirac)
"""
struct MonolayerhBN{T<:Real} <: GeneralTwoBand{T} 
    Δ::T
    t::T
    a::T
end

isperiodic(::MonolayerhBN) = true

function MonolayerhBN(
    us::UnitScaling,
    gap::Unitful.Energy=u"3.92eV",
    hopping::Unitful.Energy=u"2.33eV",
    latticeconst::Unitful.Length=u"2.51Å")

    return MonolayerhBN(
        energyscaled(gap,us),
        energyscaled(hopping,us),
        lengthscaled(latticeconst,us))
end

hx(h::MonolayerhBN,kx,ky)    = h.t * (1 + 2cos(kx * h.a/2) * cos(√3 * ky * h.a / 2))
hx(h::MonolayerhBN)          = quote $(h.t) * (1 + 2cos(kx * $(h.a)/2) * cos(√3 * ky * $(h.a) / 2)) end

hy(h::MonolayerhBN,kx,ky)    = -2h.t * cos(kx * h.a/2) * sin(√3 * ky * h.a / 2)
hy(h::MonolayerhBN)          = quote -2 * $(h.t) * cos(kx * $(h.a)/2) * sin(√3 * ky * $(h.a) / 2) end

hz(h::MonolayerhBN,kx,ky)    = h.Δ / 2
hz(h::MonolayerhBN)          = quote $(h.Δ / 2) end

dhdkx(h::MonolayerhBN,kx,ky) = SA[
    -h.t * sin(kx * h.a/2) * cos(√3 * ky * h.a / 2),
    h.t * sin(kx * h.a/2) * sin(√3 * ky * h.a / 2),
    zero(h.Δ)]
dhdkx(h::MonolayerhBN)       = SA[
    :(-$(h.t) * sin(kx*$(h.a)/2) * cos(√3*ky*$(h.a)/2)),  
    :($(h.t) * sin(kx*$(h.a)/2) * sin(√3*ky*$(h.a)/2)),  
    zero(h.Δ)]

dhdky(h::MonolayerhBN,kx,ky) = SA[
    -√3 * h.t * cos(kx * h.a/2) * sin(√3 * ky * h.a / 2),       
    -√3 * h.t * cos(kx * h.a/2) * cos(√3 * ky * h.a / 2),      
    zero(h.Δ)]
dhdky(h::MonolayerhBN)       = SA[
    :(-√3 * $(h.t) * cos(kx*$(h.a)/2) * sin(√3*ky*$(h.a)/2)),   
    :(-√3 * $(h.t) * cos(kx*$(h.a)/2) * cos(√3*ky*$(h.a)/2)),  
    zero(h.Δ)]

# Jacobian ∂h_i/∂k_j
jac(h::MonolayerhBN,kx,ky) = SA[
    -h.t * sin(kx * h.a/2) * cos(√3 * ky * h.a / 2)   -√3 * h.t * cos(kx * h.a/2) * sin(√3 * ky * h.a / 2)
    h.t * sin(kx * h.a/2) * sin(√3 * ky * h.a / 2)    -√3 * h.t * cos(kx * h.a/2) * cos(√3 * ky * h.a / 2)
    zero(h.Δ)                                           zero(h.Δ)]

jac(h::MonolayerhBN) = @SMatrix [
    :(-$(h.t) * sin(kx*$(h.a)/2) * cos(√3*ky*$(h.a)/2))     :(-√3 * $(h.t) * cos(kx*$(h.a)/2) * sin(√3*ky*$(h.a)/2))
    :($(h.t) * sin(kx*$(h.a)/2) * sin(√3*ky*$(h.a)/2))      :(-√3 * $(h.t) * cos(kx*$(h.a)/2) * cos(√3*ky*$(h.a)/2))
    quote zero($(h.Δ)) end                                  quote zero($(h.Δ)) end]


function printparamsSI(h::MonolayerhBN,us::UnitScaling;digits=3)

    tc      = timescaleSI(us)
    lc      = lengthscaleSI(us)
    Δ       = energySI(h.Δ,us)
    a       = lengthSI(h.a,us)
    t       = energySI(h.t,us)

    return """
        Δ  = $(round(typeof(Δ),Δ,sigdigits=digits)) ($(h.Δ))
        a  = $(round(typeof(a),a,sigdigits=digits)) ($(h.a))
        t  = $(round(typeof(t),t,sigdigits=digits)) ($(h.t))"""
end

getshortname(h::MonolayerhBN)    = "MonolayerhBN"
getparams(h::MonolayerhBN)       = (Δ=h.Δ,t=h.t,a=h.a)

function getparamsSI(h::MonolayerhBN,us::UnitScaling)
    Δ       = energySI(h.Δ,us)
    a       = lengthSI(h.a,us)
    t       = energySI(h.t,us)
    return (Δ=Δ,t=t,a=a)
end

gethvec(h::MonolayerhBN) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[t*(1+2cos(kx*a/2)*cos(√3*ky*a/2)), -2*t*cos(kx*a/2)*sin(√3*ky*a/2), Δ/2]
end

getdhdkx(h::MonolayerhBN) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[-t*sin(kx*a/2)*cos(√3*ky*a/2), t*sin(kx*a/2)*sin(√3*ky*a/2), zero(Δ)]
end

getdhdky(h::MonolayerhBN) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[-√3*t*cos(kx*a/2)*sin(√3*ky*a/2), -√3*t*cos(kx*a/2)*cos(√3*ky*a/2), zero(Δ)]
end

# Jacobian ∂h_i/∂k_j
getjac(h::MonolayerhBN) = let Δ=h.Δ,a=h.a,t=h.t
    (kx,ky) -> SA[
        -t*sin(kx*a/2)*cos(√3*ky*a/2)   -√3*t*cos(kx*a/2)*sin(√3*ky*a/2)
        t*sin(kx*a/2)*sin(√3*ky*a/2)    -√3*t*cos(kx*a/2)*cos(√3*ky*a/2)
        zero(Δ)                         zero(Δ)]
end

