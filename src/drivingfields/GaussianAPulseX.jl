
export efieldx
export efieldy
export GaussianAPulseX
export vecpotx
export vecpoty

"""
    GaussianAPulseX{T<:Real}

Represents spacially homogeneous, linearly polarized pulse with Gaussian envelope. 

# Mathematical form
The form of the vector potential is given by
```math
\\vec{A}(t) = \\vec{A}_0 \\cos(\\omega t + \\theta) e^{-t^2 / 2\\sigma^2}

# See also
[`GaussianAPulse`](@ref)
"""
struct GaussianAPulseX{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    θ::T
    GaussianAPulseX{T}(σ,ω,eE,θ) where {T<:Real} = new(σ,ω,eE,θ)
end
GaussianAPulseX(σ::T,ω::T,eE::T,θ::T) where {T<:Real} = GaussianAPulseX{T}(σ,ω,eE,θ)
function GaussianAPulseX(σ::Real,ω::Real,eE::Real,θ::Real=0)     
    return GaussianAPulseX(promote(σ,ω,eE,θ)...)
end
function GaussianAPulseX(us::UnitScaling,
                    standard_dev::Unitful.Time,
                    frequency::Unitful.Frequency,
                    fieldstrength::Unitful.EField,
                    θ=0)
    
    tc  = timescaleSI(us)
    lc  = lengthscaleSI(us)
    σ   = uconvert(Unitful.NoUnits,standard_dev/tc)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*tc)
    eE  = uconvert(Unitful.NoUnits,q_e*tc*lc*fieldstrength/ħ)
    return GaussianAPulseX(promote(σ,ω,eE,θ)...)
end

function efieldx(df::GaussianAPulseX)
    c1 = df.eE / (df.ω*df.σ^2)
    c2 = df.σ^2*df.ω
    return  :($c1 *(t*cos($(df.ω)*t+$(df.θ)) + $c2 * sin($(df.ω)*t+$(df.θ))) * gauss(t,$(df.σ)))
end

function vecpotx(df::GaussianAPulseX)
    c1 = df.eE / df.ω
    return :($c1 * cos($(df.ω)*t+$(df.θ)) * gauss(t,$(df.σ)))
end

efieldy(df::GaussianAPulseX) = zero(df.eE)
vecpoty(df::GaussianAPulseX) = zero(df.eE)

function efieldx(df::GaussianAPulseX,t::Real)
    return df.eE * (t*cos(df.ω*t + df.θ) + df.σ^2*df.ω*sin(df.ω*t + df.θ)) * 
        gauss(t,df.σ) / (df.ω*df.σ^2) 
end

function vecpotx(df::GaussianAPulseX,t::Real)
    return df.eE * cos(df.ω*t + df.θ) * gauss(t,df.σ) / df.ω
end

efieldy(::GaussianAPulseX,t::Real) = zero(t)
vecpoty(::GaussianAPulseX,t::Real) = zero(t)

# Specialized methods for efficiency
maximum_vecpot(df::GaussianAPulseX)  = abs(df.eE) / df.ω
maximum_vecpotx(df::GaussianAPulseX) = abs(df.eE) / df.ω
maximum_vecpoty(df::GaussianAPulseX) = zero(df.eE)

maximum_efield(df::GaussianAPulseX)  = abs(df.eE) 
maximum_efieldx(df::GaussianAPulseX) = abs(df.eE) 
maximum_efieldy(df::GaussianAPulseX) = zero(df.eE) 

central_angular_frequency(df::GaussianAPulseX) = df.ω