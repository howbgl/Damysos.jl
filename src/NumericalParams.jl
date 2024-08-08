
export getkxsamples
export getkysamples
export getnkx
export getnky
export getnt
export gettsamples
export gettspan
export NumericalParams1d
export NumericalParams2d
export NumericalParamsSingleMode

const DEFAULT_ATOL = 1e-12
const DEFAULT_RTOL = 1e-10

gettsamples(p::NumericalParameters)     = -abs(p.t0):p.dt:abs(p.t0)
getnt(p::NumericalParameters)           = length(gettsamples(p))
gettspan(p::NumericalParameters)        = (gettsamples(p)[1],gettsamples(p)[end])
getnkx(p::NumericalParameters)          = length(getkxsamples(p))
getnky(p::NumericalParameters)          = length(getkysamples(p))

for func = (:getnt,:gettsamples,:gettspan,:getkxsamples,:getnkx,:getkysamples,:getnky)
    @eval(Damysos,$func(s::Simulation) = $func(s.numericalparams))
end
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
    rtol::Real=DEFAULT_RTOL,
    atol::Real=DEFAULT_ATOL)  
    return NumericalParams2d(promote(dkx,dky,kxmax,kymax,dt,t0,rtol,atol)...)
end

NumericalParams2d(p::Dict) = construct_type_from_dict(NumericalParams2d,p)

getdimension(::NumericalParams2d) = UInt8(2)

function getparams(p::NumericalParams2d)
    return (
        dkx=p.dkx,
        dky=p.dky,
        kxmax=p.kxmax,
        kymax=p.kymax,
        dt=p.dt,
        t0=p.t0,
        rtol=p.rtol,
        atol=p.atol,
        nkx=getnkx(p),
        nky=getnky(p),
        nt=getnt(p),
        tsamples=gettsamples(p),
        tspan=gettspan(p),
        kxsamples=getkxsamples(p),
        kysamples=getkysamples(p))
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

getkysamples(p::NumericalParams2d)      = -p.kymax:p.dky:p.kymax

struct NumericalParams1d{T<:Real} <: NumericalParameters{T}
    dkx::T
    kxmax::T
    ky::T
    dt::T
    t0::T
    rtol::T
    atol::T
end

function NumericalParams1d(dkx::Real,kxmax::Real,ky::Real,dt::Real,t0::Real,
    rtol::Real=DEFAULT_RTOL,
    atol::Real=DEFAULT_ATOL)
    return NumericalParams1d(promote(dkx,kxmax,ky,dt,t0,rtol,atol)...)
end

NumericalParams1d(p::Dict) = construct_type_from_dict(NumericalParams1d,p)

getdimension(::NumericalParams1d) = UInt8(1)

function getparams(p::NumericalParams1d)
    return (
        dkx=p.dkx,
        kxmax=p.kxmax,
        kymax=p.ky,
        ky=p.ky,
        dt=p.dt,
        t0=p.t0,
        rtol=p.rtol,
        atol=p.atol,
        nkx=getnkx(p),
        nky=getnky(p),
        nt=getnt(p),
        tsamples=gettsamples(p),
        tspan=gettspan(p),
        kxsamples=getkxsamples(p),
        kysamples=getkysamples(p))
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


function getkxsamples(p::Union{NumericalParams1d,NumericalParams2d})    
    @debug @show p.dkx,p.kxmax,typeof(p.dkx),typeof(p.kxmax)
    return -p.kxmax:p.dkx:p.kxmax
end


struct NumericalParamsSingleMode{T<:Real} <: NumericalParameters{T}
    kx::T
    ky::T
    dt::T
    t0::T
    rtol::T
    atol::T
end

function NumericalParamsSingleMode(
    kx::Real,
    ky::Real,
    dt::Real,
    t0::Real,
    rtol::Real=DEFAULT_RTOL,
    atol::Real=DEFAULT_ATOL)
    args = promote(
        kx,
        ky,
        dt,
        t0,
        rtol,
        atol)
    return NumericalParamsSingleMode(args...)
end


NumericalParamsSingleMode(p::Dict) = construct_type_from_dict(NumericalParamsSingleMode,p)


getdimension(::NumericalParamsSingleMode) = UInt8(0)

function getparams(p::NumericalParamsSingleMode)
    return (
        kx=p.kx,
        ky=p.ky,
        kxmax=p.kx,
        kymax=p.ky,
        kxsamples=[p.kx],
        kysamples=[p.ky],
        dt=p.dt,
        t0=p.t0,
        rtol=p.rtol,
        atol=p.atol,
        tsamples=gettsamples(p),
        tspan=gettspan(p),
        nt=getnt(p))
end


function printparamsSI(p::NumericalParamsSingleMode,us::UnitScaling;digits=3)
    
    pnt     = getparams(p)
    kx      = wavenumberSI(p.kx,us)
    ky      = wavenumberSI(p.ky,us)
    t0      = timeSI(p.t0,us)
    dt      = timeSI(p.dt,us)
    rtol    = pnt.rtol
    atol    = pnt.atol
    nt      = getparams(p).nt
    
    symbols     = [:kx,:ky,:t0,:dt,:nt,:rtol,:atol]
    valuesSI    = [kx,ky,t0,dt,nt,rtol,atol]
    values      = [getproperty(pnt,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end

getkxsamples(p::NumericalParamsSingleMode) = [p.kx]
getkysamples(p::Union{NumericalParams1d,NumericalParamsSingleMode}) = [p.ky]
