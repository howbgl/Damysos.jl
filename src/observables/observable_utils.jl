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
        field_data = deepcopy([getproperty(x,n) for x in odata])

        upsample!(field_data)
        data,err =  Richardson.extrapolate([(d,h) for (d,h) in zip(field_data,hdata)];
            kwargs...)

        push!(timeseries, data)
        push!(errs, err)
    end

    return (O(timeseries...),errs)
end


sig(x)                      = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
bzmask1d(k,dk,kmin,kmax)    = sig((k-kmin)/(2dk)) * sig((kmax-k)/(2dk))


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
    ay = vecpoty(sim.drivingfield)
    dkx = sim.grid.kgrid.dkx
    dky = sim.grid.kgrid.dky

    if sim.drivingfield isa GaussianAPulseX # ∀times ay == 0 
        return @eval (p,t) -> bzmask1d(p[1] - $ax,$dkx,$(bz[1]),$(bz[2]))
    else
        return @eval (p,t) -> bzmask1d(p[1] - $ax,$dkx,$(bz[1]),$(bz[2])) *
           bzmask1d(p[2] - $ay,$dky,$(bz[3]),$(bz[4])) 
    end    
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

getbzbounds(::DrivingField,::KGrid0d) = ()

function getbzbounds(df::DrivingField,g::CartesianKGrid1d)
    axmax   = maximum_vecpotx(df)
    kxmax   = maximum(getkxsamples(g))
    return (-kxmax + 1.3axmax,kxmax - 1.3axmax)
end

function getbzbounds(df::DrivingField,g::Union{CartesianKGrid2d,CartesianKGrid2dStrips})
    bz_1d = getbzbounds(df,CartesianKGrid1d(g.dkx,g.kxmax))
    aymax   = maximum_vecpoty(df)
    kymax   = maximum(getkysamples(g))
    return (bz_1d...,-kymax + 1.3aymax,kymax - 1.3aymax)
end


function printBZSI(df::DrivingField,g::CartesianKGrid2d,us::UnitScaling;digits=3)
    
    bz    = getbzbounds(df,g)
    bzSI  = [wavenumberSI(k,us) for k in bz]
    bzSI  = map(x -> round(typeof(x),x,sigdigits=digits),bzSI)
    bz    = [round(x,sigdigits=digits) for x in bz]

    return """
        BZ(kx) = [$(bzSI[1]),$(bzSI[2])] ([$(bz[1]),$(bz[2])])
        BZ(ky) = [$(bzSI[3]),$(bzSI[4])] ([$(bz[3]),$(bz[4])])\n"""
end

function printBZSI(df::DrivingField,g::CartesianKGrid2dStrips,us::UnitScaling;digits=3)
    kymin = wavenumberSI(g.kymin,us)
    t     = typeof(kymin)
    str   = printBZSI(df,CartesianKGrid2d(g.dkx,g.kxmax,g.dky,g.kymax),us;digits=digits)
    str   *= "BZ patched together with previous one at $(round(t,kymin,sigdigits=digits))"
    return str
end

function printBZSI(df::DrivingField,g::CartesianKGrid1d,us::UnitScaling;digits=3)
    
    bz    = getbzbounds(df,g)
    bzSI  = [wavenumberSI(k,us) for k in bz]
    bzSI  = map(x -> round(typeof(x),x,sigdigits=digits),bzSI)
    bz    = [round(x,sigdigits=digits) for x in bz]

    return """
        BZ(kx) = [$(bzSI[1]),$(bzSI[2])] ([$(bz[1]),$(bz[2])])\n"""
end

function printBZSI(df::DrivingField,g::KGrid0d,us::UnitScaling;digits=3)
    return ""
end