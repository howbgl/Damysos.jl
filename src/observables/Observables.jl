import LinearAlgebra: normalize!,copyto!
import Base: +,-,*,zero,empty

export Observable,getnames_obs,zero!,resize

sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 

include("Velocity.jl")
include("Occupation.jl")

function getmovingbz(sim::Simulation)

    kxs = getkxsamples(sim.numericalparams)
    kys = getkysamples(sim.numericalparams) 

    return getmovingbz(sim,kxs,kys)
end

function getmovingbz(
    sim::Simulation,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    bz  = getbzbounds(sim)
    nkx = length(kxsamples)
    nky = length(kysamples)
    dkx = sim.numericalparams.dkx
    dky = sim.numericalparams.dky
    ts  = gettsamples(sim.numericalparams)
    ax  = get_vecpotx(sim.drivingfield)
    ay  = get_vecpoty(sim.drivingfield)

    moving_bz       = zeros(eltype(kxs),nkx,nky,length(ts))
    bzmask1d(kx)    = sig((kx-bz[1])/(2dkx)) * sig((bz[2]-kx)/(2dkx))

    for (i,t) in enumerate(ts)
        moving_bz[:,i] .= bzmask1d.(kxsamples .- ax(t))
    end

    return moving_bz
end

function getmovingbz(sim::Simulation,kxsamples::AbstractVector{<:Real}) 
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

function integrateobs_threaded!(
    observables::Vector{Vector{Observable{T}}},
    observables_dest::Vector{Observable{T}},
    vertices::AbstractVector{T}) where {T<:Real}

    for (i,odest) in enumerate(observables_dest) 
        integrateobs_threaded!([o[i] for o in observables],odest,collect(vertices))
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

function sumobservables_ktile!(
    sim::Simulation,
    sols,
    ktile,
    moving_bz,
    obsfuncs)
    
    for (o,funcs) in zip(sim.observables,obsfuncs)
        sumobservable_ktile!(sim,o,sols,ktile,moving_bz,funcs)
    end
end