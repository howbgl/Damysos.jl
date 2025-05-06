export gauss
export get_efieldx
export get_efieldy
export get_vecpotx
export get_vecpoty
export scaledriving_frequency

gauss(t::T,σ::T) where {T<:Real} = exp(-t^2 / (2σ^2))

function scaledriving_frequency(ufrequency,ufermivelocity)
    return scaledriving_frequency(promote(ufrequency,ufermivelocity)...)
end

function scaledriving_frequency(
        frequency::Unitful.Frequency{T},
        fermivelocity::Unitful.Velocity{T}) where{T<:Real}

    tc = uconvert(u"fs",1/frequency)
    lc = uconvert(u"nm",fermivelocity*tc)
    return UnitScaling(tc,lc)
end

# General fallback methods, more specialized functions can be provided for efficiency
maximum_vecpot(df::DrivingField) = maximum((maximum_vecpotx(df),maximum_vecpoty(df)))
maximum_efield(df::DrivingField) = maximum((maximum_efieldx(df),maximum_efieldy(df)))

function maximum_vecpotx(df::DrivingField) 
    dt = 2π / (df.ω * 128)
    ts = -5df.σ:dt:5df.σ
    return maximum([vecpotx(df,t) for t in ts])
end

function maximum_vecpoty(df::DrivingField) 
    dt = 2π / (df.ω * 128)
    ts = -5df.σ:dt:5df.σ
    return maximum([vecpoty(df,t) for t in ts])
end

function maximum_efieldx(df::DrivingField) 
    dt = 2π / (df.ω * 128)
    ts = -5df.σ:dt:5df.σ
    return maximum([efieldx(df,t) for t in ts])
end

function maximum_efieldy(df::DrivingField) 
    dt = 2π / (df.ω * 128)
    ts = -5df.σ:dt:5df.σ
    return maximum([efieldy(df,t) for t in ts])
end


central_frequency(df::DrivingField)     = central_angular_frequency(df) / (2π)
function central_frequency_SI(df::DrivingField,us::UnitScaling)
    return frequencySI(central_frequency(df),us)
end

# TODO generic fallback for central frequency (via Fourier trafo?)


get_efieldx(sim::Simulation) = get_efieldx(sim.drivingfield)
get_efieldy(sim::Simulation) = get_efieldy(sim.drivingfield)
get_vecpotx(sim::Simulation) = get_vecpotx(sim.drivingfield)
get_vecpoty(sim::Simulation) = get_vecpoty(sim.drivingfield)

include("GaussianAPulse.jl")
include("GaussianEPulse.jl")
include("CompositeDrivingField.jl")
