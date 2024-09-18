
export GaussianEPulse_s

"""
    GaussianEPulse_s{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the electric field is given by
```math
\\vec{E}(t) = \\vec{E}_0 \\sin(\\omega t+\\phi) e^{-t^2 / \\sigma^2}
``` 
where ``\\vec{E}_0=E_0(\\cos\\varphi\\,\\vec{e}_x + \\sin\\varphi\\,\\vec{e}_y``). 

"""
struct GaussianEPulse_s{T<:Real} <: DrivingField{T}
    σ::T
    ω_1::T
    ω_2::T
    eE_1::T
    eE_2::T
    ϕ::T
end
function GaussianEPulse_s(σ::Real,ω_1::Real,ω_2::Real,eE_1::Real,eE_2::Real,ϕ::Real=0)
    GaussianEPulse_s(promote(σ,ω_1,ω_2,eE_1,eE_2,ϕ)...)
end
function GaussianEPulse_s(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency_1::Unitful.Frequency,
                    frequency_2::Unitful.Frequency,
                    fieldstrength_1::Unitful.EField,
                    fieldstrength_2::Unitful.EField,
                    ϕ=0)
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω_1   = uconvert(Unitful.NoUnits,2π*frequency_1*p.timescale)
    ω_2   = uconvert(Unitful.NoUnits,2π*frequency_2*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE_1  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength_1/Unitful.ħ)
    eE_2  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength_2/Unitful.ħ)
    return GaussianEPulse_s(promote(σ,ω_1,ω_2,eE_1,eE_2,ϕ)...)
end

function getparams(df::GaussianEPulse_s)
    return (σ=df.σ,ν_1=df.ω_1/2π,ν_2=df.ω_2/2π,ω_1=df.ω_1,ω_2=df.ω_2,eE_1=df.eE_1,eE_2=df.eE_2,ħω_1=df.ω_1,ħω2=df.ω_2,ϕ=df.ϕ)
end

function get_efieldx(df::GaussianEPulse_s)
    return let σ=df.σ,eE_1=df.eE_1,eE_2=df.eE_2,ω_1=df.ω_1,ω_2=df.ω_2,ϕ=df.ϕ
        t -> (eE_1*cos(ω_1*t+ϕ) + eE_2*cos(ω_2*t+ϕ)) * gauss(t,σ)
    end 
end

function get_efieldy(df::GaussianEPulse_s)
    return let σ=df.σ,eE_1=df.eE_1,eE_2=df.eE_2,ω_1=df.ω_1,ω_2=df.ω_2,ϕ=df.ϕ
        t -> (eE_1*sin(ω_1*t+ϕ) - eE_2*sin(ω_2*t+ϕ)) * gauss(t,σ)
    end   
end

function get_vecpotx(df::GaussianEPulse_s)
    return let
        t ->  -im*0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2))*exp(-im*df.ϕ) *
               ( df.eE_1*exp(df.σ^2*df.ω_2^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ)) - (exp(2*im*df.ϕ)*erfi( (im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ))))
               + df.eE_2*exp(df.σ^2*df.ω_1^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)) - (exp(2*im*df.ϕ)*erfi( (im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)))))
    end 
end

function get_vecpoty(df::GaussianEPulse_s)
    return let
        t ->  -0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2))*exp(-im*df.ϕ) *
               (-df.eE_1*exp(df.σ^2*df.ω_2^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ)) + (exp(2*im*df.ϕ)*erfi( (im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ))))
               + df.eE_2*exp(df.σ^2*df.ω_1^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)) + (exp(2*im*df.ϕ)*erfi( (im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)))))
    end 
end

function getfields(df::GaussianEPulse_s)
    return (get_vecpotx(df),get_vecpoty(df),get_efieldx(df),get_efieldy(df))
end

function efieldx(df::GaussianEPulse_s)
    return :(($df.eE_1 * cos($(df.ω_1)*t + $(df.ϕ)) + $df.eE_2 * cos($(df.ω_2)*t + $(df.ϕ))) * exp(-t^2 / (2*$df.σ^2)))
end

function efieldx(df::GaussianEPulse_s,t::Real)
    return (df.eE_1 * cos(df.ω_1*t + df.ϕ) + df.eE_2 * cos(df.ω_2*t + df.ϕ)) * exp(-t^2 / (2df.σ^2))
end

function efieldy(df::GaussianEPulse_s)
    return :(($df.eE_1 * sin($(df.ω_1)*t + $(df.ϕ)) - $df.eE_2 * sin($(df.ω_2)*t + $(df.ϕ))) * exp(-t^2 / (2*$df.σ^2)))
end

function efieldy(df::GaussianEPulse_s,t::Real)
    return (df.eE_1 * sin(df.ω_1*t + df.ϕ) - df.eE_2 * sin(df.ω_2*t + df.ϕ)) * exp(-t^2 / (2df.σ^2))
end


function vecpotx(df::GaussianEPulse_s)
    pref   = convert(typeof(df.σ),0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2)))
    amp1   = convert(typeof(df.σ),df.eE_1*exp(df.σ^2*df.ω_2^2/2.))
    amp2   = convert(typeof(df.σ),df.eE_2*exp(df.σ^2*df.ω_1^2/2.))
    phase1 = exp(-im*df.ϕ)
    phase2 = exp(2*im*df.ϕ)
    c1     = df.σ^2 * df.ω_1
    c2     = df.σ^2 * df.ω_2
    c      = sqrt(2)*df.σ
    return :(-im*$pref*$phase1*($amp1*(erfi( (-im*t+$c1) / $c) - $phase2*erfi( (im*t+$c1) / $c))
            + $amp2*(erfi( (-im*t+$c2) / $c) - $phase2*erfi( (im*t+$c2) / $c))))
end
"""
function vecpotabs(df::GaussianEPulse_s,t::Real)
    amp     = convert(typeof(df.σ),-df.σ*df.eE*sqrt(π/2.) * exp(-df.σ^2*df.ω^2/2.))
    phase   = exp(im*df.ϕ)
    c1      = df.σ^2 * df.ω
    c2      = sqrt(2)*df.σ
    return amp * imag(phase * erf( (t-im * c1) / c2 ))
end
"""

function vecpotx(df::GaussianEPulse_s,t::Real)
    return -im*0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2))*exp(-im*df.ϕ) *
     (df.eE_1*exp(df.σ^2*df.ω_2^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ)) - (exp(2*im*df.ϕ)*erfi((im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ))))
    + df.eE_2*exp(df.σ^2*df.ω_1^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)) - (exp(2*im*df.ϕ)*erfi((im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)))))
end

function vecpoty(df::GaussianEPulse_s)
    pref   = convert(typeof(df.σ),0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2)))
    amp1   = convert(typeof(df.σ),df.eE_1*exp(df.σ^2*df.ω_2^2/2.))
    amp2   = convert(typeof(df.σ),df.eE_2*exp(df.σ^2*df.ω_1^2/2.))
    phase1 = exp(-im*df.ϕ)
    phase2 = exp(2*im*df.ϕ)
    c1     = df.σ^2 * df.ω_1
    c2     = df.σ^2 * df.ω_2
    c      = sqrt(2)*df.σ
    return :(-$pref*$phase1*(-$amp1*(erfi( (-im*t+$c1) / $c) + $phase2*erfi( (im*t+$c1) / $c))
            + $amp2*(erfi( (-im*t+$c2) / $c) + $phase2*erfi( (im*t+$c2) / $c))))
end

function vecpoty(df::GaussianEPulse_s,t::Real)
    return -0.5*df.σ*sqrt(π/2.)*exp(-0.5*df.σ^2*(df.ω_1^2+df.ω_2^2))*exp(-im*df.ϕ) *
     (-df.eE_1*exp(df.σ^2*df.ω_2^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ)) + (exp(2*im*df.ϕ)*erfi((im*t+(df.σ^2 * df.ω_1)) / (sqrt(2)*df.σ))))
     + df.eE_2*exp(df.σ^2*df.ω_1^2/2.)*(erfi((-im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)) + (exp(2*im*df.ϕ)*erfi((im*t+(df.σ^2 * df.ω_2)) / (sqrt(2)*df.σ)))))
end

function printparamsSI(df::GaussianEPulse_s,us::UnitScaling;digits=4)
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