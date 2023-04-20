

function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, 
                        range::AbstractVector{T}) where {T<:Real}

    hashstring   = sprintf1("%x",hash([sim,comp,param,range]))
    ensname      = "Ensemble[$(length(range))]{$T}($(sim.dimensions)d)" * 
            getshortname(sim.hamiltonian) * getshortname(sim.drivingfield) *
            "_$param"*"_sweep_" * hashstring

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
        sweeplist[i] = Simulation(new_h,new_df,new_p,deepcopy(sim.observables),
                sim.unitscaling,sim.dimensions,name,
                "/home/how09898/phd/data/hhgjl/" * ensname * '/' * name * '/',
                "/home/how09898/phd/plots/hhgjl/" * ensname * '/' * name * '/')
    end


    return Ensemble(sweeplist,"_$param"*"_sweep_"*hashstring,
                "/home/how09898/phd/data/hhgjl/" * ensname * '/',
                "/home/how09898/phd/plots/hhgjl/" * ensname * '/')
end


function maximum_k(df::DrivingField)
    @warn "using fallback for maximum k value of DrivingField!"
    return df.eE/df.ω
end
maximum_k(df::GaussianPulse) = df.eE/df.ω

function semiclassical_interband_range(h::GappedDirac,df::DrivingField)
    ϵ        = getϵ(h)
    ωmin     = 2.0*ϵ(0.0,0.0)
    kmax     = maximum_k(df)
    ωmax     = 2.0*ϵ(kmax,0.0)
    min_harm = ωmin/df.ω
    max_harm = ωmax/df.ω
    println("Approximate range of semiclassical interband: ",min_harm," to ",
            max_harm," (harmonic number)")
end
