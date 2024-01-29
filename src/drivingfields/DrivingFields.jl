
export gauss
export getfields
export get_efieldx
export get_efieldx_expression
export get_efieldy
export get_efieldy_expression
export get_field_expressions
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

function get_field_expressions(df::DrivingField)
    return Expr(:block,
        get_efieldx_expression(df),
        get_efieldy_expression(df),
        get_vecpotx_expression(df),
        get_vecpoty_expression(df))
end

function getfields(df::DrivingField)
    return (get_vecpotx(df),get_vecpoty(df),get_efieldx(df),get_efieldy(df))
end


include("GaussianAPulse.jl")
include("GaussianEPulse.jl")
