import LinearAlgebra: normalize!,copyto!
import Base: +,-,*,zero,empty

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


function getbzbounds(df::GaussianAPulse,p::NumericalParams1d)
    kxmax = p.kxmax
    axmax = df.eE / df.ω
    return (-kxmax + 1.3axmax,kxmax - 1.3axmax)
end

function getbzbounds(df::GaussianAPulse,p::NumericalParams2d)

    amax = 1.3df.eE / df.ω
    return (
        -p.kxmax + cos(df.φ)*amax,
        p.kxmax - cos(df.φ)*amax,
        -p.kymax + sin(df.φ)*amax,
        p.kymax - sin(df.φ)*amax)
end

function maximum_kdisplacement(df::DrivingField,ts::AbstractVector{<:Real})
    axmax = maximum(abs.(map(t -> vecpotx(df,t),ts)))
    aymax = maximum(abs.(map(t -> vecpoty(df,t),ts)))
    return maximum((axmax,aymax))
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