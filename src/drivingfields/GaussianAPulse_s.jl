
export efieldx_s
export efieldy_s
export GaussianAPulse_s
export vecpotx_s
export vecpoty_s

"""
    GaussianAPulse_s{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the vector potential is given by
```math
\\vec{A}(t) = \\vec{A}_0 \\cos(\\omega t + \\theta) e^{-t^2 / \\sigma^2}
``` 
where ``\\vec{A}_0=A_0(\\cos\\varphi\\,\\vec{e}_x + \\sin\\varphi\\,\\vec{e}_y``). 

"""
struct GaussianAPulse_s{T<:Real} <: DrivingField{T}
    σ::T
    ω_1::T
    ω_2::T
    eE_1::T
    eE_2::T
    θ::T
    GaussianAPulse_s{T}(σ,ω_1,ω_2,eE_1,eE_2,θ) where {T<:Real} = new(σ,ω_1,ω_2,eE_1,eE_2,θ)
end
GaussianAPulse_s(σ::T,ω_1::T,ω_2::T,eE_1::T,eE_2::T,θ::T) where {T<:Real} = GaussianAPulse_s{T}(σ,ω_1,ω_2,eE_1,eE_2,θ)
function GaussianAPulse_s(σ::Real,ω_1::Real,ω_2::Real,eE_1::Real,eE_2::Real,θ::Real=0)     
    return GaussianAPulse_s(promote(σ,ω_1,ω_2,eE_1,eE_2,θ)...)
end
function GaussianAPulse_s(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency_1::Unitful.Frequency,
                    frequency_2::Unitful.Frequency,
                    fieldstrength_1::Unitful.EField,
                    fieldstrength_2::Unitful.EField,
                    θ=0)
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω_1   = uconvert(Unitful.NoUnits,2π*frequency_1*p.timescale)
    ω_2   = uconvert(Unitful.NoUnits,2π*frequency_2*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE_1  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength_1/Unitful.ħ)
    eE_2  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength_2/Unitful.ħ)
    return GaussianAPulse_s(promote(σ,ω_1,ω_2,eE_1,eE_2,θ)...)
end

# type alias for backwards compatibility
export GaussianPulse_s
GaussianPulse_s = GaussianAPulse_s

function getparams(df::GaussianAPulse_s)
    return (σ=df.σ,ν_1=df.ω_1/2π,ν_2=df.ω_2/2π,ω_1=df.ω_1,ω_2=df.ω_2,eE_1=df.eE_1,eE_2=df.eE_2,ħω_1=df.ω_1,ħω_2=df.ω_2,θ=df.θ)
end

@inline function get_efieldx(df::GaussianAPulse_s)
    return t-> (df.eE_1 * (1/(df.ω_1*df.σ^2)*t*cos(df.ω_1*t + df.θ) + sin(df.ω_1*t + df.θ))
               +df.eE_2 * (1/(df.ω_2*df.σ^2)*t*cos(df.ω_2*t + df.θ) + sin(df.ω_2*t + df.θ))) * gauss(t,df.σ)  
end
@inline function get_vecpotx(df::GaussianAPulse_s)
    return t -> (df.eE_1/df.ω_1 * cos(df.ω_1*t + df.θ) + df.eE_2/df.ω_2 * cos(df.ω_2*t + df.θ)) * gauss(t,df.σ)
end

@inline function get_efieldy(df::GaussianAPulse_s)
    return t-> (df.eE_1 * (1/(df.ω_1*df.σ^2)*t*sin(df.ω_1*t + df.θ) - cos(df.ω_1*t + df.θ))
               -df.eE_2 * (1/(df.ω_2*df.σ^2)*t*sin(df.ω_2*t + df.θ) - cos(df.ω_2*t + df.θ))) * gauss(t,df.σ)
end
@inline function get_vecpoty(df::GaussianAPulse_s)
    return t -> (df.eE_1/df.ω_1 * sin(df.ω_1*t + df.θ) - df.eE_2/df.ω_2 * sin(df.ω_2*t + df.θ)) * gauss(t,df.σ)
end

function getfields(df::GaussianAPulse_s)
    return (get_vecpotx(df),get_vecpoty(df),get_efieldx(df),get_efieldy(df))
end

function efieldx(df::GaussianAPulse_s)
    c1_1 = df.eE_1 / (df.ω_1*df.σ^2)
    c1_2 = df.eE_1
    c2_1 = df.eE_2 / (df.ω_2*df.σ^2)
    c2_2 = df.eE_2
    return  :(($c1_1 *t*cos($(df.ω_1)*t+$(df.θ)) + $c1_2 * sin($(df.ω_1)*t+$(df.θ))
             + $c2_1 *t*cos($(df.ω_2)*t+$(df.θ)) + $c2_2 * sin($(df.ω_2)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function efieldy(df::GaussianAPulse_s)
    c1_1 = df.eE_1 / (df.ω_1*df.σ^2)
    c1_2 = df.eE_1
    c2_1 = df.eE_2 / (df.ω_2*df.σ^2)
    c2_2 = df.eE_2
    return  :(($c1_1 *t*sin($(df.ω_1)*t+$(df.θ)) - $c1_2 * cos($(df.ω_1)*t+$(df.θ))
             - $c2_1 *t*sin($(df.ω_2)*t+$(df.θ)) + $c2_2 * cos($(df.ω_2)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function vecpotx(df::GaussianAPulse_s)
    c1 = df.eE_1 / df.ω_1
    c2 = df.eE_2 / df.ω_2
    return :(($c1 * cos($(df.ω_1)*t+$(df.θ)) + $c2 * cos($(df.ω_2)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function vecpoty(df::GaussianAPulse_s)
    c1 = df.eE_1 / df.ω_1
    c2 = df.eE_2 / df.ω_2
    return :(($c1 * sin($(df.ω_1)*t+$(df.θ)) - $c2 * sin($(df.ω_2)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function efieldx(df::GaussianAPulse_s,t::Real)
    return (df.eE_1 * (1/(df.ω_1*df.σ^2)*t*cos(df.ω_1*t + df.θ) + sin(df.ω_1*t + df.θ))
           +df.eE_2 * (1/(df.ω_2*df.σ^2)*t*cos(df.ω_2*t + df.θ) + sin(df.ω_2*t + df.θ))) * gauss(t,df.σ)  
end

function vecpotx(df::GaussianAPulse_s,t::Real)
    return (df.eE_1/df.ω_1 * cos(df.ω_1*t + df.θ) + df.eE_2/df.ω_2 * cos(df.ω_2*t + df.θ)) * gauss(t,df.σ)
end

function efieldy(df::GaussianAPulse_s,t::Real)
    return (df.eE_1 * (1/(df.ω_1*df.σ^2)*t*sin(df.ω_1*t + df.θ) - cos(df.ω_1*t + df.θ))
           -df.eE_2 * (1/(df.ω_2*df.σ^2)*t*sin(df.ω_2*t + df.θ) - cos(df.ω_2*t + df.θ))) * gauss(t,df.σ)  
end

function vecpoty(df::GaussianAPulse_s,t::Real)
    return (df.eE_1/df.ω_1 * sin(df.ω_1*t + df.θ) - df.eE_2/df.ω_2 * sin(df.ω_2*t + df.θ)) * gauss(t,df.σ)
end

function printparamsSI(df::GaussianAPulse_s,us::UnitScaling;digits=4)

    p       = getparams(df)
    σ       = timeSI(df.σ,us)
    ω_1     = uconvert(u"fs^-1",frequencySI(df.ω_1,us))
    ω_2     = uconvert(u"fs^-1",frequencySI(df.ω_2,us))
    ħω_1    = uconvert(u"eV",energySI(df.ω_1,us))
    ħω_2    = uconvert(u"eV",energySI(df.ω_2,us))
    ν_1     = frequencySI(df.ω_1/2π,us)
    ν_2     = frequencySI(df.ω_2/2π,us)
    eE_1    = electricfieldSI(df.eE_1,us)
    eE_2    = electricfieldSI(df.eE_2,us)

    symbols     = [:σ,:ω_1,:ω_2,:ν_1,:ν_2,:eE_1,:eE_2,:ħω_1,ħω_2]
    valuesSI    = [σ,ω_1,ω_2,ν_1,ν_2,eE_1,eE_2,ħω_1,ħω_2]
    values      = [getproperty(p,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end