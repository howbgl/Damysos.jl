import LinearAlgebra: normalize!,copyto!
import Base: +,-,*,zero,empty,isapprox

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

function buildbzmask_expression_upt(sim::Simulation)

    bz = getbzbounds(sim)
    ax = vecpotx(sim.drivingfield)
    dkx = sim.numericalparams.dkx

    return :(bzmask1d(p[1] - $ax,$dkx,$(bz[1]),$(bz[2])))
end

function buildbzmask_expression(sim::Simulation)

    bz = getbzbounds(sim)
    ax = vecpotx(sim.drivingfield)
    dkx = sim.numericalparams.dkx

    return :(bzmask1d(kx - $ax,$dkx,$(bz[1]),$(bz[2])))
end

function buildobservable_expression_upt(sim::Simulation)
    expressions = [buildobservable_expression_upt(sim,o) for o in sim.observables]
    return :([$(expressions...)])
end

function buildobservable_expression(sim::Simulation)
    expressions = [buildobservable_expression(sim,o) for o in sim.observables]
    return :([$(expressions...)])
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
