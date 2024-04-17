

export UnitScaling

export electricfield_scaled
export electricfieldSI
export energyscaled
export energySI
export frequencyscaled
export frequencySI
export getparams
export lengthscaled
export lengthSI
export timescaled
export timeSI
export velocityscaled
export velocitySI
export wavenumberscaled
export wavenumberSI


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
See [here](https://en.wikipedia.org/w/index.php?title=Nondimensionalization&oldid=1166582079)
"""
struct UnitScaling{T<:Real} <: SimulationComponent{T}
    timescale::T
    lengthscale::T
end
function UnitScaling(timescale,lengthscale) 
    return UnitScaling(ustrip(u"fs",timescale),ustrip(u"nm",lengthscale))
end
function getparams(us::UnitScaling{T}) where {T<:Real} 
    return (timescale=Quantity(us.timescale,u"fs"),
            lengthscale=Quantity(us.lengthscale,u"nm"))
end

function energySI(en::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"meV",en*Unitful.ħ/tc)
end

function energyscaled(energy::Unitful.Energy,us::UnitScaling)
    tc,lc = getparams(us)
    ħ     = Unitful.ħ
    return uconvert(Unitful.NoUnits,tc*energy/ħ)
end

function electricfieldSI(field::Real,us::UnitScaling)
    tc,lc   = getparams(us)
    e       = uconvert(u"C",1u"eV"/1u"V")
    ħ       = Unitful.ħ
    return uconvert(u"MV/cm",field*ħ/(e*tc*lc))
end

function electricfield_scaled(field::Unitful.EField,us::UnitScaling)
    tc,lc   = getparams(us)
    e       = uconvert(u"C",1u"eV"/1u"V")
    ħ       = Unitful.ħ
    return uconvert(Unitful.NoUnits,e*tc*lc*field/ħ)
end

function timeSI(time::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"fs",time*tc)
end

function timescaled(time::Unitful.Time,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,time/tc)    
end

function lengthSI(length::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å",length*lc)
end
function lengthscaled(length::Unitful.Length,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,length/lc)
end

function frequencySI(ν::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"THz",ν/tc)
end

function frequencyscaled(ν::Unitful.Frequency,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,ν*tc)
end

function velocitySI(v::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"m/s",v*lc/tc)
end

function velocityscaled(v::Unitful.Velocity,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,v*tc/lc)
end

function wavenumberSI(k::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å^-1",k/lc)
end

function wavenumberscaled(k::Unitful.Wavenumber,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,k*lc)
end
