struct NumericalParams2d{T<:Real} <: NumericalParameters{T}
    dkx::T
    dky::T
    kxmax::T
    kymax::T
    dt::T
    t0::T
    rtol::T
    atol::T
end
function NumericalParams2d(dkx::Real,dky::Real,kxmax::Real,kymax::Real,dt::Real,t0::Real,
                            rtol::Real,atol::Real)  
    return NumericalParams2d(promote(dkx,dky,kxmax,kymax,dt,t0,rtol,atol)...)
end
function NumericalParams2d(dkx::Real,dky::Real,kxmax::Real,kymax::Real,dt::Real,t0::Real)
    return NumericalParams2d(dkx,dky,kxmax,kymax,dt,t0,1e-12,1e-12)
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
    rtol=p.rtol,
    atol=p.atol,
    nkx=2*Int(cld(p.kxmax,p.dkx)),
    nky=2*Int(cld(p.kymax,p.dky)),
    nt=2*Int(cld(abs(p.t0),p.dt)),
    tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
    kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))),
    kysamples=LinRange(-p.kymax,p.kymax,2*Int(cld(p.kymax,p.dky))))
end

function printparamsSI(p::NumericalParams2d,us::UnitScaling;digits=3)

    pnt     = getparams(p)
    kxmax   = wavenumberSI(p.kxmax,us)
    dkx     = wavenumberSI(p.dkx,us)
    nkx     = getparams(p).nkx
    kymax   = wavenumberSI(p.kymax,us)
    dky     = wavenumberSI(p.dky,us)
    nky     = getparams(p).nky
    t0      = timeSI(p.t0,us)
    dt      = timeSI(p.dt,us)
    rtol    = pnt.rtol
    atol    = pnt.atol
    nt      = pnt.nt

    symbols     = [:kxmax,:dkx,:nkx,:kymax,:dky,:nky,:t0,:dt,:rtol,:atol,:nt]
    valuesSI    = [kxmax,dkx,nkx,kymax,dky,nky,t0,dt,rtol,atol,nt]
    values      = [getproperty(pnt,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end


struct NumericalParams1d{T<:Real} <: NumericalParameters{T}
    dkx::T
    kxmax::T
    dt::T
    t0::T
    rtol::T
    atol::T
end
function NumericalParams1d(dkx::Real,kxmax::Real,dt::Real,t0::Real,rtol::Real,atol::Real)     
    return NumericalParams1d(promote(dkx,kxmax,dt,t0,rtol,atol)...)
end
function NumericalParams1d(dkx::Real,kxmax::Real,dt::Real,t0::Real)
    return NumericalParams1d(dkx,kxmax,dt,t0,1e-12,1e-12)
end

function getparams(p::NumericalParams1d{T}) where {T<:Real} 
    return (
    dkx=p.dkx,
    kxmax=p.kxmax,
    dt=p.dt,
    t0=p.t0,
    rtol=p.rtol,
    atol=p.atol,
    nkx=2*Int(cld(p.kxmax,p.dkx)),
    nt=2*Int(cld(abs(p.t0),p.dt)),
    tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
    kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))))
end

function printparamsSI(p::NumericalParams1d,us::UnitScaling;digits=3)

    pnt     = getparams(p)
    kxmax   = wavenumberSI(p.kxmax,us)
    dkx     = wavenumberSI(p.dkx,us)
    nkx     = getparams(p).nkx
    t0      = timeSI(p.t0,us)
    dt      = timeSI(p.dt,us)
    rtol    = pnt.rtol
    atol    = pnt.atol
    nt      = getparams(p).nt
    
    symbols     = [:kxmax,:dkx,:nkx,:t0,:dt,:nt,:rtol,:atol]
    valuesSI    = [kxmax,dkx,nkx,t0,dt,nt,rtol,atol]
    values      = [getproperty(pnt,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end
