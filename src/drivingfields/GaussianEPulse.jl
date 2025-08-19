
export GaussianEPulse

"""
    GaussianEPulse{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the electric field is given by
```math
\\vec{E}(t) = \\vec{E}_0 \\sin(\\omega t+\\phi) e^{-t^2 / \\sigma^2}
``` 
where ``\\vec{E}_0=E_0(\\cos\\varphi\\,\\vec{e}_x + \\sin\\varphi\\,\\vec{e}_y``). 

"""
struct GaussianEPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    φ::T
    ϕ::T
end
function GaussianEPulse(σ::Real,ω::Real,eE::Real,φ::Real=0,ϕ::Real=0)
    GaussianEPulse(promote(σ,ω,eE,φ,ϕ)...)
end
function GaussianEPulse(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency::Unitful.Frequency,
                    fieldstrength::Unitful.EField,
                    φ=0,
                    ϕ=0)
    tc  = timescaleSI(us)
    lc  = lengthscaleSI(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/tc)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*tc)
    eE  = uconvert(Unitful.NoUnits,q_e*tc*lc*fieldstrength/ħ)
    return GaussianEPulse(promote(σ,ω,eE,φ,ϕ)...)
end

function getparams(df::GaussianEPulse)
    return (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE,φ=df.φ,ħω=df.ω,ϕ=df.ϕ)
end

function efieldx(df::GaussianEPulse)
    σ22 = 2df.σ^2
    amp = cos(df.φ)*df.eE
    return :($amp * sin($(df.ω)*t + $(df.ϕ)) * exp(-t^2 / $σ22))
end

function efieldx(df::GaussianEPulse,t::Real)
    return cos(df.φ)*df.eE * sin(df.ω*t + df.ϕ) * exp(-t^2 / (2df.σ^2))
end

function efieldy(df::GaussianEPulse)
    σ22 = 2df.σ^2
    amp = sin(df.φ)*df.eE
    return :($amp * sin($(df.ω)*t + $(df.ϕ)) * exp(-t^2 / $σ22))
end

function efieldy(df::GaussianEPulse,t::Real)
    return sin(df.φ)*df.eE * sin(df.ω*t + df.ϕ) * exp(-t^2 / (2df.σ^2))
end

function vecpotx(df::GaussianEPulse)
    amp    = convert(typeof(df.σ),-df.σ*df.eE*cos(df.φ)*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    phase  = exp(im*df.ϕ)
    c1     = df.σ^2 * df.ω
    c2     = sqrt(2)*df.σ
    return :($amp * imag($phase * erf( (t-im * $c1) / $c2 )))
end

function vecpotabs(df::GaussianEPulse,t::Real)
    amp     = convert(typeof(df.σ),-df.σ*df.eE*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    phase   = exp(im*df.ϕ)
    c1      = df.σ^2 * df.ω
    c2      = sqrt(2)*df.σ
    return amp * imag(phase * erf( (t-im * c1) / c2 ))
end

function vecpotx(df::GaussianEPulse,t::Real)
    return cos(df.φ) * vecpotabs(df,t)
end

function vecpoty(df::GaussianEPulse)
    amp    = convert(typeof(df.σ),-df.σ*df.eE*sin(df.φ)*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    phase  = exp(im*df.ϕ)
    c1     = df.σ^2 * df.ω
    c2     = sqrt(2)*df.σ
    return :($amp * imag($phase * erf( (t-im * $c1) / $c2 )))
end

function vecpoty(df::GaussianEPulse,t::Real)
    return sin(df.φ) * vecpotabs(df,t)
end

central_angular_frequency(df::GaussianEPulse) = df.ω

function printparamsSI(df::GaussianEPulse,us::UnitScaling;digits=4)

    σ       = timeSI(df.σ,us)
    ω       = uconvert(u"fs^-1",frequencySI(df.ω,us))
    ħω      = uconvert(u"eV",energySI(df.ω,us))
    ν       = frequencySI(df.ω/2π,us)
    eE      = electricfieldSI(df.eE,us)
    φ       = df.φ

    symbols     = [:σ,:ω,:ν,:eE,:φ,:ħω]
    valuesSI    = [σ,ω,ν,eE,φ,ħω]
    values      = [df.σ,df.ω,central_frequency(df),df.eE,df.φ,df.ω]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end