
export GaussianAPulse

"""
    GaussianAPulse{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the vector potential is given by
```math
\\vec{A}(t) = \\vec{A}_0 \\cos(\\omega t) e^{-t^2 / \\sigma^2}
``` 
where ``\\vec{A}_0=A_0(\\cos\\varphi\\,\\vec{e}_x + \\sin\\varphi\\,\\vec{e}_y``). 

"""
struct GaussianAPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    φ::T
    GaussianAPulse{T}(σ,ω,eE,φ) where {T<:Real} = new(σ,ω,eE,φ)
end
GaussianAPulse(σ::T,ω::T,eE::T,φ::T) where {T<:Real} = GaussianAPulse{T}(σ,ω,eE,φ)
GaussianAPulse(σ::Real,ω::Real,eE::Real,φ::Real)     = GaussianAPulse(promote(σ,ω,eE,φ)...)
GaussianAPulse(σ::Real,ω::Real,eE::Real)             = GaussianAPulse(σ,ω,eE,0)
function GaussianAPulse(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency::Unitful.Frequency,
                    fieldstrength::Unitful.EField,
                    φ=0)
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength/Unitful.ħ)
    return GaussianAPulse(promote(σ,ω,eE,φ)...)
end

# type alias for backwards compatibility
export GaussianPulse
GaussianPulse = GaussianAPulse

function getparams(df::GaussianAPulse{T}) where {T<:Real}  
    return (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE,φ=df.φ,ħω=df.ω)
end

@inline function get_efieldx(df::GaussianAPulse{T}) where {T<:Real}
    return t-> cos(df.φ) * df.eE * (t*cos(df.ω*t) + df.σ^2*df.ω*sin(df.ω*t)) * 
                gauss(t,df.σ) / (df.ω*df.σ^2)  
end
@inline function get_vecpotx(df::GaussianAPulse{T}) where {T<:Real}
    return t -> cos(df.φ) * df.eE * cos(df.ω*t) * gauss(t,df.σ) / df.ω
end

@inline function get_efieldy(df::GaussianAPulse{T}) where {T<:Real}
    return t-> sin(df.φ) * df.eE * (t*cos(df.ω*t) + df.σ^2*df.ω*sin(df.ω*t)) * 
                gauss(t,df.σ) / (df.ω*df.σ^2)  
end
@inline function get_vecpoty(df::GaussianAPulse{T}) where {T<:Real}
    return t -> sin(df.φ) * df.eE * cos(df.ω*t) * gauss(t,df.σ) / df.ω
end

function getfields(df::GaussianAPulse)
    return (get_vecpotx(df),get_vecpoty(df),get_efieldx(df),get_efieldy(df))
end

function printparamsSI(df::GaussianAPulse,us::UnitScaling;digits=4)

    p       = getparams(df)
    σ       = timeSI(df.σ,us)
    ω       = uconvert(u"fs^-1",frequencySI(df.ω,us))
    ħω      = uconvert(u"eV",energySI(df.ω,us))
    ν       = frequencySI(df.ω/2π,us)
    eE      = electricfieldSI(df.eE,us)
    φ       = df.φ

    symbols     = [:σ,:ω,:ν,:eE,:φ,:ħω]
    valuesSI    = [σ,ω,ν,eE,φ,ħω]
    values      = [getproperty(p,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end

function get_efieldx_expression(df::GaussianAPulse)
    return :(fx(t) = $(cos(df.φ) * df.eE)*(t*cos($(df.ω)*t)+$(df.σ^2*df.ω)*sin($(df.ω)*t))* 
        gauss(t,$(df.σ)) / $(df.ω*df.σ^2))
end

function get_efieldy_expression(df::GaussianAPulse)
    return :(fy(t) = $(sin(df.φ) * df.eE)*(t*cos($(df.ω)*t)+$(df.σ^2*df.ω)*sin($(df.ω)*t))* 
        gauss(t,$(df.σ)) / $(df.ω*df.σ^2))
end

function get_vecpotx_expression(df::GaussianAPulse)
    return :(ax(t) =  cos($(df.ω)*t) * gauss(t,$(df.σ)) *$(cos(df.φ) * df.eE / df.ω))
end

function get_vecpoty_expression(df::GaussianAPulse)
    return :(ay(t) =  cos($(df.ω)*t) * gauss(t,$(df.σ)) *$(sin(df.φ) * df.eE / df.ω))
end

function makedefining_expression_efieldx(df::GaussianAPulse,name=:fx)

    warning_quote   = warn_ifdefined_quote(name)
    defining_quote  = :($name(t) = $(cos(df.φ) * df.eE)*(t*cos($(df.ω)*t)+$(df.σ^2*df.ω)*
        sin($(df.ω)*t)) * gauss(t,$(df.σ)) / $(df.ω*df.σ^2))
    return Expr(:block,warning_quote,defining_quote)
end

function makedefining_expression_efieldy(df::GaussianAPulse,name=:fy)

    warning_quote   = warn_ifdefined_quote(name)
    defining_quote  = :($name(t) = $(sin(df.φ) * df.eE)*(t*cos($(df.ω)*t)+$(df.σ^2*df.ω)*
        sin($(df.ω)*t))*gauss(t,$(df.σ)) / $(df.ω*df.σ^2))
    return Expr(:block,warning_quote,defining_quote)
end

function makedefining_expression_vecpotx(df::GaussianAPulse,name=:ax)

    warning_quote   = warn_ifdefined_quote(name)
    defining_quote  = :($name(t) =  cos($(df.ω)*t) * gauss(t,$(df.σ)) *
        $(cos(df.φ) * df.eE / df.ω))
    return Expr(:block,warning_quote,defining_quote)
end

function makedefining_expression_vecpoty(df::GaussianAPulse,name=:ay)

    warning_quote   = warn_ifdefined_quote(name)
    defining_quote  = :($name(t) =  cos($(df.ω)*t) * gauss(t,$(df.σ)) *
        $(sin(df.φ) * df.eE / df.ω))
    return Expr(:block,warning_quote,defining_quote)
end

