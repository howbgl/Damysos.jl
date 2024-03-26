import LinearAlgebra: normalize!,copyto!
import Base: +,-,*,zero,empty

export buildbzmask
export buildbzmask_expression
export buildobservable
export buildobservable_expression
export bzmask1d
export getnames_obs
export Observable
export observable_from_data
export resize
export sig
export zero!


sig(x)                      = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
bzmask1d(kx,dkx,kmin,kmax)  = sig((kx-kmin)/(2dkx)) * sig((kmax-kx)/(2dkx))

include("Velocity.jl")
include("Occupation.jl")

function buildbzmask(sim::Simulation)
    expr = buildbzmask_expression(sim)
    return @eval (kx,ky,t) -> $expr
end

function buildbzmask_expression(sim::Simulation)

    bz = getbzbounds(sim)
    ax = vecpotx(sim.drivingfield)
    dkx = sim.numericalparams.dkx

    return :(bzmask1d(kx - $ax,$dkx,$(bz[1]),$(bz[2])))
end

function buildobservable(sim::Simulation)
    return @eval (u,p,t) -> $(buildobservable_expression(sim))
end

function buildobservable_expression(sim::Simulation)
    expressions = [buildobservable_expression(sim,o) for o in sim.observables]
    return :([$(expressions...)])
end

function observable_from_data(sim::Simulation,data)

    observabledata = [empty([d]) for d in data[1]]
    @show observabledata
    for slice in data
        for (d,obs) in zip(slice,observabledata) 
            push!(obs,d)
        end
    end
    return observabledata
end

function getmovingbz(sim::Simulation{T}) where {T<:Real}
    p              = getparams(sim)
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

function integrateobs!(
    observables::Vector{Vector{Observable{T}}},
    observables_dest::Vector{Observable{T}},
    vertices::AbstractVector{T}) where {T<:Real}

    for (i,odest) in enumerate(observables_dest) 
        integrateobs!([o[i] for o in observables],odest,collect(vertices))
    end    
end

function integrateobs_kxbatch_add!(
    sim::Simulation{T},
    sol,
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    obsfuncs) where {T<:Real}

    for (o,funcs) in zip(sim.observables,obsfuncs)
        integrateobs_kxbatch_add!(sim,o,sol,kxsamples,ky,moving_bz,funcs)
    end        
end