
gauss(t::T,σ::T) where {T<:Real} = exp(-t^2 / (2σ^2))

getshortname(df::DrivingField{T}) where {T<:Real} = split("_$df",'{')[1]

struct GaussianPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    φ::T
    GaussianPulse{T}(σ,ω,eE,φ) where {T<:Real} = new(σ,ω,eE,φ)
end
GaussianPulse(σ::T,ω::T,eE::T,φ::T) where {T<:Real} = GaussianPulse{T}(σ,ω,eE,φ)
GaussianPulse(σ::Real,ω::Real,eE::Real,φ::Real)     = GaussianPulse(promote(σ,ω,eE,φ)...)
GaussianPulse(σ::Real,ω::Real,eE::Real)             = GaussianPulse(σ,ω,eE,0)
function GaussianPulse(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency::Unitful.Frequency,
                    fieldstrength::Unitful.EField,
                    φ=0)
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength/Unitful.ħ)
    return GaussianPulse(promote(σ,ω,eE,φ)...)
end

function getparams(df::GaussianPulse{T}) where {T<:Real}  
    return (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE,φ=df.φ)
end

@inline function get_efieldx(df::GaussianPulse{T}) where {T<:Real}
    return t-> cos(df.φ) * df.eE * (t*cos(df.ω*t) + df.σ^2*df.ω*sin(df.ω*t)) * 
                gauss(t,df.σ) / (df.ω*df.σ^2)  
end
@inline function get_vecpotx(df::GaussianPulse{T}) where {T<:Real}
    return t -> cos(df.φ) * df.eE * cos(df.ω*t) * gauss(t,df.σ) / df.ω
end

@inline function get_efieldy(df::GaussianPulse{T}) where {T<:Real}
    return t-> sin(df.φ) * df.eE * (t*cos(df.ω*t) + df.σ^2*df.ω*sin(df.ω*t)) * 
                gauss(t,df.σ) / (df.ω*df.σ^2)  
end
@inline function get_vecpoty(df::GaussianPulse{T}) where {T<:Real}
    return t -> sin(df.φ) * df.eE * cos(df.ω*t) * gauss(t,df.σ) / df.ω
end

function printparamsSI(df::GaussianPulse,us::UnitScaling;digits=3)

    p       = getparams(df)
    σ       = timeSI(df.σ,us)
    ω       = uconvert(u"fs^-1",frequencySI(df.ω,us))
    ν       = frequencySI(df.ω/2π,us)
    eE      = electricfieldSI(df.eE,us)
    φ       = df.φ

    symbols     = [:σ,:ω,:ν,:eE,:φ]
    valuesSI    = [σ,ω,ν,eE,φ]
    values      = [getproperty(p,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end

get_efieldx(sim::Simulation{T}) where {T<:Real} = get_efieldx(sim.drivingfield)
get_efieldy(sim::Simulation{T}) where {T<:Real} = get_efieldy(sim.drivingfield)
get_vecpotx(sim::Simulation{T}) where {T<:Real} = get_vecpotx(sim.drivingfield)
get_vecpoty(sim::Simulation{T}) where {T<:Real} = get_vecpoty(sim.drivingfield)
