
struct GaussianPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    GaussianPulse{T}(σ,ω,eE) where {T<:Real} = new(σ,ω,eE)
end
GaussianPulse(σ::T,ω::T,eE::T) where {T<:Real} = GaussianPulse{T}(σ,ω,eE)
GaussianPulse(σ::Real,ω::Real,eE::Real)        = GaussianPulse(promote(σ,ω,eE)...)
function GaussianPulse(us::UnitScaling{T},standard_dev::Unitful.Time{T},
                    frequency::Unitful.Frequency{T},fieldstrength::Unitful.EField{T}) where{T<:Real}
    p   = getparams(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/p.timescale)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*p.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE  = uconvert(Unitful.NoUnits,e*p.timescale*p.lengthscale*fieldstrength/Unitful.ħ)
    return GaussianPulse(σ,ω,eE)
end
GaussianPulse(us,ustandard_dev,ufrequency,ufieldstrength) = GaussianPulse(us,promote(ustandard_dev,ufrequency,ufieldstrength)...)

getparams(df::GaussianPulse{T}) where {T<:Real}  = (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE)

function get_efield(df::GaussianPulse{T}) where {T<:Real}
    return t-> df.eE * (t * cos(df.ω*t) + df.σ^2 * df.ω * sin(df.ω*t)) * exp(-t^2 / (2df.σ^2)) / (df.ω*df.σ^2) 
end
function get_vecpot(df::GaussianPulse{T}) where {T<:Real}
    return t -> df.eE * cos(df.ω*t) * exp(-t^2 / (2df.σ^2)) / df.ω
end

