
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
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength/Unitful.ħ)
    return GaussianEPulse(promote(σ,ω,eE,φ,ϕ)...)
end

function getparams(df::GaussianEPulse)
    return (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE,φ=df.φ,ħω=df.ω,ϕ=df.ϕ)
end

@inline function get_efieldx(df::GaussianEPulse)
    return let σ=df.σ,eE=df.eE,ω=df.ω,φ=df.φ,ϕ=df.ϕ
        t -> cos(φ)*eE*sin(ω*t+ϕ)*gauss(t,σ)
    end 
end

@inline function get_efieldy(df::GaussianEPulse)
    return let σ=df.σ,eE=df.eE,ω=df.ω,φ=df.φ,ϕ=df.ϕ
        t -> sin(φ)*eE*sin(ω*t+ϕ)*gauss(t,σ)
    end   
end

@inline function get_vecpotx(df::GaussianEPulse)
    factor1 = convert(typeof(df.σ),-df.σ*df.eE*cos(df.φ)*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    factor2 = exp(im*df.ϕ)
    return let σ=df.σ,ω=df.ω,a=factor1,b=factor2
        t -> a*imag(b*erf( (t-im*σ^2*ω) / (sqrt(2)*σ) ))
    end 
end

@inline function get_vecpoty(df::GaussianEPulse)
    factor1 = convert(typeof(df.σ),-df.σ*df.eE*sin(df.φ)*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    factor2 = exp(im*df.ϕ)
    return let σ=df.σ,ω=df.ω,a=factor1,b=factor2
        t -> a*imag(b*erf( (t-im*σ^2*ω) / (sqrt(2)*σ) ))
    end 
end

function getfields(df::GaussianEPulse)
    return (get_vecpotx(df),get_vecpoty(df),get_efieldx(df),get_efieldy(df))
end

function printparamsSI(df::GaussianEPulse,us::UnitScaling;digits=4)

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