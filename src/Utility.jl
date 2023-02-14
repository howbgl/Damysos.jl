
struct UnitScaling{T<:Real}
    timescale::Unitful.Time{T}
    lengthscale::Unitful.Length{T}
end
getparams(us::UnitScaling{T}) where {T<:Real} = (timescale=us.timescale,lengthscale=us.lengthscale)


function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, range::AbstractVector{T}) where {T<:Real}

    ensname     = "_$param"*"_sweep"
    ensdirname  = lowercase("Ensemble[$(length(range))]{$T}($(sim.dimensions)d)" * split("_$(sim.hamiltonian)",'{')[1] * split("_$(sim.drivingfield)",'{')[1]) * ensname

    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    for i in eachindex(sweeplist)
        name = "$param=$(range[i])"
        if comp isa Hamiltonian{T}
            new_h  = set(comp,PropertyLens(param),range[i])
            new_df = sim.drivingfield
            new_p  = sim.numericalparams
        elseif comp isa DrivingField{T}
            new_h  = sim.hamiltonian
            new_df = set(comp,PropertyLens(param),range[i])
            new_p  = sim.numericalparams
        elseif comp isa NumericalParameters{T}
            new_h  = sim.hamiltonian
            new_df = sim.drivingfield
            new_p  = set(comp,PropertyLens(param),range[i])
        else
            return nothing
        end
        sweeplist[i] = Simulation(new_h,new_df,new_p,sim.observables,
                        sim.unitscaling,sim.dimensions,name,
                        sim.datapath * ensdirname * '/' * name * '/',
                        sim.plotpath * ensdirname * '/' * name * '/')
    end

    return Ensemble(sweeplist,ensname)
end


function semiclassical_interband_range(h::GappedDirac,df::GaussianPulse)
    ϵ        = getϵ(h)
    ωmin     = 2.0*ϵ(0.0,0.0)
    kmax     = df.eE/df.ω
    ωmax     = 2.0*ϵ(kmax,0.0)
    min_harm = ωmin/df.ω
    max_harm = ωmax/df.ω
    println("Approximate range of semiclassical interband: ",min_harm," to ",
            max_harm," (harmonic number)")
end
