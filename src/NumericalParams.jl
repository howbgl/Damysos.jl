
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


struct NumericalParams2dSlice{T<:Real} <: NumericalParameters{T}
    params::NumericalParams2d{T}
    kyspan::Tuple{T,T}
end

function getparams(p::NumericalParams2dSlice{T}) where {T<:Real}
    fullparams  = getparams(p.params)
    kys_full    = collect(fullparams.kysamples)
    kysamples   = kys_full[kys_full .>= p.kyspan[1] .&& kys_full .<= p.kyspan[2]]
    nky         = length(kysamples)
    return merge(fullparams,(nky=nky,kysamples=kysamples,))
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

function printparamsSI(p::NumericalParams2d,us::UnitScaling;digits=3)
    kxmax   = wavenumberSI(p.kxmax,us)
    dkx     = wavenumberSI(p.dkx,us)
    nkx     = getparams(p).nkx
    kymax   = wavenumberSI(p.kymax,us)
    dky     = wavenumberSI(p.dky,us)
    nky     = getparams(p).nky
    t0      = timeSI(p.t0,us)
    dt      = timeSI(p.dt,us)
    nt      = getparams(p).nt
    str = "kxmax  = $(round(typeof(kxmax),kxmax,sigdigits=digits))\n"
    str *= "dkx  = $(round(typeof(dkx),dkx,sigdigits=digits))\n"
    str *= "nkx  = $nkx\n"
    str *= "kymax  = $(round(typeof(kymax),kymax,sigdigits=digits))\n"
    str *= "dky  = $(round(typeof(dky),dky,sigdigits=digits))\n"
    str *= "nky  = $nky\n"
    str *= "t0 = $(round(typeof(t0),t0,sigdigits=digits))\n"
    str *= "dt = $(round(typeof(dt),dt,sigdigits=digits))\n"
    str *= "nt  = $nt\n"
    return str
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

function printparamsSI(p::NumericalParams1d,us::UnitScaling;digits=3)
    kxmax   = wavenumberSI(p.kxmax,us)
    dkx     = wavenumberSI(p.dkx,us)
    nkx     = getparams(p).nkx
    t0      = timeSI(p.t0,us)
    dt      = timeSI(p.dt,us)
    nt      = getparams(p).nt
    str = "kxmax  = $(round(typeof(kxmax),kxmax,sigdigits=digits))\n"
    str *= "dkx  = $(round(typeof(dkx),dkx,sigdigits=digits))\n"
    str *= "nkx  = $nkx\n"
    str *= "t0 = $(round(typeof(t0),t0,sigdigits=digits))\n"
    str *= "dt = $(round(typeof(dt),dt,sigdigits=digits))\n"
    str *= "nt  = $nt\n"
    return str
end
