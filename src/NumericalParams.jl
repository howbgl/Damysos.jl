
struct NumericalParams2d{T<:Real} <: NumericalParameters{T}
    dkx::T
    dky::T
    kxmax::T
    kymax::T
    dt::T
    t0::T
    function NumericalParams2d{T}(dkx,dky,kxmax,kymax,dt,t0) where {T<:Real} 
        return new(dkx,dky,kxmax,kymax,dt,t0)
    end
end
function NumericalParams2d(dkx::T,dky::T,kxmax::T,kymax::T,dt::T,t0::T) where {T<:Real}    
    return NumericalParams2d{T}(dkx,dky,kxmax,kymax,dt,t0)
end
function NumericalParams2d(dkx::Real,dky::Real,kxmax::Real,kymax::Real,dt::Real,t0::Real)  
    return NumericalParams2d(promote(dkx,dky,kxmax,kymax,dt,t0)...)
end

function getparams(p::NumericalParams2d{T}) where {T<:Real} 
    return (
    dkx=p.dkx,
    dky=p.dky,
    kxmax=p.kxmax,
    kymax=p.kymax,
    dt=p.dt,
    t0=p.t0,
    nkx=2*Int(cld(p.kxmax,p.dkx)),
    nky=2*Int(cld(p.kymax,p.dky)),
    nt=2*Int(cld(abs(p.t0),p.dt)),
    tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
    kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))),
    kysamples=LinRange(-p.kymax,p.kymax,2*Int(cld(p.kymax,p.dky))))
end

struct NumericalParams1d{T<:Real} <: NumericalParameters{T}
    dkx::T
    kxmax::T
    dt::T
    t0::T
    NumericalParams1d{T}(dkx,kxmax,dt,t0) where{T<:Real} = new(dkx,kxmax,dt,t0)
end

function NumericalParams1d(dkx::T,kxmax::T,dt::T,t0::T) where {T<:Real} 
    return NumericalParams1d{T}(dkx,kxmax,dt,t0)
end

function NumericalParams1d(dkx::Real,kxmax::Real,dt::Real,t0::Real)     
    return NumericalParams1d(promote(dkx,kxmax,dt,t0)...)
end

function getparams(p::NumericalParams1d{T}) where {T<:Real} 
    return (
    dkx=p.dkx,
    kxmax=p.kxmax,
    nkx=2*Int(cld(p.kxmax,p.dkx)),
    nt=2*Int(cld(abs(p.t0),p.dt)),
    dt=p.dt,
    tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
    kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))))
end
