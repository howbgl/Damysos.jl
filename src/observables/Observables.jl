import LinearAlgebra: normalize!,copyto!
import Base: +,-,*,zero,empty

export bzmask1d
export getnames_obs
export Observable
export observable_from_data
export resize
export sig
export zero!

Vector{Observable}(obs::Observable...)               = Observable[obs...]
Vector{Observable}(::SimulationComponent{T}) where T = Observable{T}[]

function Base.isapprox(
    o1::Vector{Observable{T}},
    o2::Vector{Observable{U}};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    return Base.isapprox.(o1,o2;atol=atol,rtol=rtol,nans=nans) |> all
end

function count_nans(obs::Vector{<:Observable})
	return sum(count_nans.(obs))
end

function count_nans(o::Observable)
    n = 0
    for s in fieldnames(typeof(o))
        n += sum(isnan.(getproperty(o,s)))
    end
    return n
end


function extrapolate(obs_h_itr::AbstractVector{<:Tuple{<:Observable{T}, <:Number}};
    invert_h = false,
    kwargs...) where T

    oh_itr      = filter(x -> count_nans(x[1]) == 0,obs_h_itr) 
    odata       = first.(oh_itr)
    O           = eltype(odata)
    hdata       = invert_h ? [1/oh[2] for oh in oh_itr] : last.(oh_itr)
    timeseries  = []
    errs        = Vector{T}(undef,0)

    isempty(oh_itr) && return (obs_h_itr[end][1],fill(Inf,fieldcount(O)))

    for n in fieldnames(O)
        field_data = [getproperty(x,n) for x in odata]

        upsample!(field_data)
        data,err =  Richardson.extrapolate([(d,h) for (d,h) in zip(field_data,hdata)];
            kwargs...)

        push!(timeseries, data)
        push!(errs, err)
    end

    return (O(timeseries...),errs)
end


sig(x)                      = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
bzmask1d(kx,dkx,kmin,kmax)  = sig((kx-kmin)/(2dkx)) * sig((kmax-kx)/(2dkx))

include("Velocity.jl")
include("Occupation.jl")

function timesplit_obs(obs::Vector{<:Observable},ts::Vector{<:Vector{<:Real}})
    return [[resize(o,length(t)) for o in obs] for t in ts]
end

function timemerge_obs(obsvec::Vector{<:Vector{<:Observable}})
    firstobs = obsvec[1]
    for obs in obsvec[2:end]
        for (o1,o2) in zip(firstobs,obs)
            append!(o1,o2)
        end        
    end
    return firstobs
end

function define_bzmask(sim::Simulation)

    bz = getbzbounds(sim)
    ax = vecpotx(sim.drivingfield)
    dkx = sim.numericalparams.dkx

    @eval (p,t) -> bzmask1d(p[1] - $ax,$dkx,$(bz[1]),$(bz[2]))
end

function write_ensemblesols_to_observables!(sim::Simulation,data)

    resize_obs!(sim)
    observabledata = [empty([d]) for d in data[1]]
    for slice in data
        for (d,obs) in zip(slice,observabledata) 
            push!(obs,d)
        end
    end
    for (o,d) in zip(sim.observables,observabledata)
        write_ensembledata_to_observable!(o,d)
    end
    return sim.observables
end

function getmovingbz(sim::Simulation)
    p = getparams(sim)
    return getmovingbz(sim,p.kxsamples)
end

function getmovingbz(sim::Simulation{T},kxsamples::AbstractVector{T}) where {T<:Real}
    p              = getparams(sim)
    nkx            = length(kxsamples)
    ax             = get_vecpotx(sim.drivingfield)
    ay             = get_vecpoty(sim.drivingfield)
    moving_bz      = zeros(T,nkx,p.nt)
    bzmask1d(kx)   = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))

    for i in 1:p.nt
        moving_bz[:,i] .= bzmask1d.(kxsamples .- ax(p.tsamples[i]))
    end
    return moving_bz    
end


getbzbounds(::DrivingField,::NumericalParamsSingleMode) = ()

function getbzbounds(df::DrivingField,p::NumericalParams1d)
    axmax   = maximum_vecpotx(df)
    kxmax   = maximum(getkxsamples(p))
    return (-kxmax + 1.3axmax,kxmax - 1.3axmax)
end

function getbzbounds(df::DrivingField,p::NumericalParams2d)
    bz_1d = getbzbounds(df,NumericalParams1d(p.dkx,p.kxmax,0.0,p.dt,p.t0,p.rtol,p.atol))
    aymax   = maximum_vecpoty(df)
    kymax   = maximum(getkysamples(p))
    return (bz_1d...,-kymax + 1.3aymax,kymax - 1.3aymax)
end


function printBZSI(df::DrivingField,p::NumericalParams2d,us::UnitScaling;digits=3)
    
    bz    = getbzbounds(df,p)
    bzSI  = [wavenumberSI(k,us) for k in bz]
    bzSI  = map(x -> round(typeof(x),x,sigdigits=digits),bzSI)
    bz    = [round(x,sigdigits=digits) for x in bz]

    return """
        BZ(kx) = [$(bzSI[1]),$(bzSI[2])] ([$(bz[1]),$(bz[2])])
        BZ(ky) = [$(bzSI[3]),$(bzSI[4])] ([$(bz[3]),$(bz[4])])\n"""
end

function printBZSI(df::DrivingField,p::NumericalParams1d,us::UnitScaling;digits=3)
    
    bz    = getbzbounds(df,p)
    bzSI  = [wavenumberSI(k,us) for k in bz]
    bzSI  = map(x -> round(typeof(x),x,sigdigits=digits),bzSI)
    bz    = [round(x,sigdigits=digits) for x in bz]

    return """
        BZ(kx) = [$(bzSI[1]),$(bzSI[2])] ([$(bz[1]),$(bz[2])])\n"""
end

function printBZSI(df::DrivingField,p::NumericalParamsSingleMode,us::UnitScaling;digits=3)
    return ""
end