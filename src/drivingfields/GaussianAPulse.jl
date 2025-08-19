
export efieldx
export efieldy
export GaussianAPulse
export vecpotx
export vecpoty

"""
    GaussianAPulse{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the vector potential is given by
```math
\\vec{A}(t) = \\vec{A}_0 \\cos(\\omega t + \\theta) e^{-t^2 / 2\\sigma^2}
``` 
where ``\\vec{A}_0=A_0(\\cos\\varphi\\,\\vec{e}_x + \\sin\\varphi\\,\\vec{e}_y``). 

# See also
[`GaussianAPulseX`](@ref)
"""
struct GaussianAPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    φ::T
    θ::T
    GaussianAPulse{T}(σ,ω,eE,φ,θ) where {T<:Real} = new(σ,ω,eE,φ,θ)
end
GaussianAPulse(σ::T,ω::T,eE::T,φ::T,θ::T) where {T<:Real} = GaussianAPulse{T}(σ,ω,eE,φ,θ)
function GaussianAPulse(σ::Real,ω::Real,eE::Real,φ::Real=0,θ::Real=0)     
    return GaussianAPulse(promote(σ,ω,eE,φ,θ)...)
end
function GaussianAPulse(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency::Unitful.Frequency,
                    fieldstrength::Unitful.EField,
                    φ=0,
                    θ=0)
    
    tc  = timescaleSI(us)
    lc  = lengthscaleSI(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/tc)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*tc)
    eE  = uconvert(Unitful.NoUnits,q_e*tc*lc*fieldstrength/ħ)
    return GaussianAPulse(promote(σ,ω,eE,φ,θ)...)
end

# type alias for backwards compatibility
export GaussianPulse
GaussianPulse = GaussianAPulse

function efieldx(df::GaussianAPulse)
    c1 = cos(df.φ) * df.eE / (df.ω*df.σ^2)
    c2 = df.σ^2*df.ω
    return  :($c1 *(t*cos($(df.ω)*t+$(df.θ)) + $c2 * sin($(df.ω)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function efieldy(df::GaussianAPulse)
    c1 = sin(df.φ) * df.eE / (df.ω*df.σ^2)
    c2 = df.σ^2*df.ω
    return  :($c1 *(t*cos($(df.ω)*t+$(df.θ)) + $c2 * sin($(df.ω)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function vecpotx(df::GaussianAPulse)
    c1 = cos(df.φ) * df.eE / df.ω
    return :($c1 * cos($(df.ω)*t+$(df.θ)) * gauss(t,$(df.σ)))
end

function vecpoty(df::GaussianAPulse)
    c1 = sin(df.φ) * df.eE / df.ω
    return :($c1 * cos($(df.ω)*t+$(df.θ)) * gauss(t,$(df.σ)))
end

efieldx(df::GaussianAPulse,t::Real) = cos(df.φ) * efieldx(GaussianAPulseX(df),t)
vecpotx(df::GaussianAPulse,t::Real) = cos(df.φ) * vecpotx(GaussianAPulseX(df),t)
efieldy(df::GaussianAPulse,t::Real) = sin(df.φ) * efieldy(GaussianAPulseX(df),t)
vecpoty(df::GaussianAPulse,t::Real) = sin(df.φ) * vecpoty(GaussianAPulseX(df),t)

# Specialized methods for efficiency
maximum_vecpot(df::GaussianAPulse)  = abs(df.eE) / df.ω
maximum_vecpotx(df::GaussianAPulse) = cos(df.φ) * abs(df.eE) / df.ω
maximum_vecpoty(df::GaussianAPulse) = sin(df.φ) * abs(df.eE) / df.ω

maximum_efield(df::GaussianAPulse)  = abs(df.eE) 
maximum_efieldx(df::GaussianAPulse) = cos(df.φ) * abs(df.eE) 
maximum_efieldy(df::GaussianAPulse) = sin(df.φ) * abs(df.eE) 

central_angular_frequency(df::GaussianAPulse) = df.ω

function printparamsSI(df::Union{GaussianAPulse,GaussianAPulseX},us::UnitScaling;digits=4)

    σ       = timeSI(df.σ,us)
    ω       = uconvert(u"fs^-1",frequencySI(df.ω,us))
    ħω      = uconvert(u"eV",energySI(df.ω,us))
    ν       = frequencySI(df.ω/2π,us)
    eE      = electricfieldSI(df.eE,us)

    symbols     = [:σ,:ω,:ν,:eE,:ħω]
    valuesSI    = [σ,ω,ν,eE,ħω]
    values      = [df.σ,df.ω,central_frequency(df),df.eE,df.ω]
    str         = ""

    if df isa GaussianAPulse
        push!(symbols,:φ)
        push!(valuesSI,df.φ)
        push!(values,df.φ)
    end

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end