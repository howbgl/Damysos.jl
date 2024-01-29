
export gauss
export get_efieldx
export get_efieldx_expression
export get_efieldy
export get_efieldy_expression
export get_vecpotx
export get_vecpotx_expression
export get_vecpoty
export get_vecpoty_expression
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

include("GaussianAPulse.jl")
include("GaussianEPulse.jl")
