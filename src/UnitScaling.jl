

export UnitScaling

export q_e
export m_e
export ħ
export electricfield_scaled
export electricfieldSI
export energyscaled
export energySI
export frequencyscaled
export frequencySI
export lengthscaled
export lengthSI
export timescaled
export timeSI
export velocityscaled
export velocitySI
export wavenumberscaled
export wavenumberSI

"Elementary charge as Quantity (Unitful.jl package) equal to 1.602176634e-19C"
const q_e = u"1.602176634e-19C"

"Rest mass of an electron as Quantity (Unitful.jl package) equal to 9.1093837139(28)e-31kg"
const m_e = u"9.1093837139e-31kg"

"Reduced Planck constant as Quantity (Unitful.jl package) equal to 6.582119569...e-16 eV⋅s"
const ħ = Unitful.ħ

"""
    UnitScaling(timescale,lengthscale)

Represents a physical length- and time-scale used for non-dimensionalization of a system.

# Examples
```jldoctest
julia> using Unitful; us = UnitScaling(u"1.0s",u"1.0m")
UnitScaling:
 timescale: 1.0e15 fs
 lengthscale: 1.0e9 nm


```

# Further information
Internally, the fields timescale & lengthscale of UnitScaling are saved in femtoseconds 
and nanometers, but never used for numerical calculations. They are only needed to convert
to dimensionful quantities again (Unitful package used, supports SI,cgs,... units)
See [here](https://en.wikipedia.org/w/index.php?title=Nondimensionalization&oldid=1166582079)
for more information on non-dimensionalization.
"""
struct UnitScaling{T<:Real} <: SimulationComponent{T}
    timescale::T
    lengthscale::T
end
function UnitScaling(timescale,lengthscale) 
    return UnitScaling(ustrip(u"fs",timescale),ustrip(u"nm",lengthscale))
end

timescaleSI(us::UnitScaling)    = Quantity(us.timescale,u"fs")
lengthscaleSI(us::UnitScaling)  = Quantity(us.lengthscale,u"nm")


function energySI(en::Real,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(u"meV",en*Unitful.ħ/tc)
end

function energyscaled(energy::Unitful.Energy,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(Unitful.NoUnits,tc*energy/ħ)
end

function electricfieldSI(field::Real,us::UnitScaling)
    tc = timescaleSI(us)
    lc = lengthscaleSI(us)
    return uconvert(u"MV/cm",field*ħ/(q_e*tc*lc))
end

function electricfield_scaled(field::Unitful.EField,us::UnitScaling)
    tc = timescaleSI(us)
    lc = lengthscaleSI(us)
    return uconvert(Unitful.NoUnits,q_e*tc*lc*field/ħ)
end

function timeSI(time::Real,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(u"fs",time*tc)
end

function timescaled(time::Unitful.Time,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(Unitful.NoUnits,time/tc)    
end

function lengthSI(length::Real,us::UnitScaling)
    lc = lengthscaleSI(us)
    return uconvert(u"Å",length*lc)
end
function lengthscaled(length::Unitful.Length,us::UnitScaling)
    lc = lengthscaleSI(us)
    return uconvert(Unitful.NoUnits,length/lc)
end

function frequencySI(ν::Real,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(u"THz",ν/tc)
end

function frequencyscaled(ν::Unitful.Frequency,us::UnitScaling)
    tc = timescaleSI(us)
    return uconvert(Unitful.NoUnits,ν*tc)
end

function velocitySI(v::Real,us::UnitScaling)
    tc = timescaleSI(us)
    lc = lengthscaleSI(us)
    return uconvert(u"m/s",v*lc/tc)
end

function velocityscaled(v::Unitful.Velocity,us::UnitScaling)
    tc = timescaleSI(us)
    lc = lengthscaleSI(us)
    return uconvert(Unitful.NoUnits,v*tc/lc)
end

function wavenumberSI(k::Real,us::UnitScaling)
    lc = lengthscaleSI(us)
    return uconvert(u"Å^-1",k/lc)
end

function wavenumberscaled(k::Unitful.Wavenumber,us::UnitScaling)
    lc = lengthscaleSI(us)
    return uconvert(Unitful.NoUnits,k*lc)
end
