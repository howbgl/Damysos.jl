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


get_efieldx(sim::Simulation) = get_efieldx(sim.drivingfield)
get_efieldy(sim::Simulation) = get_efieldy(sim.drivingfield)
get_vecpotx(sim::Simulation) = get_vecpotx(sim.drivingfield)
get_vecpoty(sim::Simulation) = get_vecpoty(sim.drivingfield)

include("GaussianAPulse.jl")
include("GaussianEPulse.jl")
include("GaussianAPulse_s.jl")
include("GaussianEPulse_s.jl")