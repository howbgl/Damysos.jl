function stringexpand_vector(v::AbstractVector)
    str = ""
    for s in v
        str *= "$(s)_"
    end
    return str[1:end-1] # drop last underscore
end

droplast(path::AbstractString) = joinpath(splitpath(path)[1:end-1]...)

function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, 
                        range::AbstractVector{T}) where {T<:Real}

    return parametersweep(sim,comp,[param],[[r] for r in range])
end

function parametersweep(sim::Simulation{T},comp::SimulationComponent{T},
    params::Vector{Symbol},range::Vector{Vector{T}}) where {T<:Real}

    hashstring   = sprintf1("%x",hash([sim,comp,params,range]))
    ensname      = "Ensemble[$(length(range))]{$T}($(sim.dimensions)d)" * 
            getshortname(sim.hamiltonian) * getshortname(sim.drivingfield) * "_" *
            stringexpand_vector(params)*"_sweep_" * hashstring

    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    for i in eachindex(sweeplist)

        name = ""
        for (p,v) in zip(params,range[i])
            name *= "$p=$(v)_"
        end
        name = name[1:end-1] # drop last underscore

        new_h  = deepcopy(sim.hamiltonian)
        new_df = deepcopy(sim.drivingfield)
        new_p  = deepcopy(sim.numericalparams)

        if comp isa Hamiltonian{T}
            for (p,v) in zip(params,range[i])
                new_h  = set(new_h,PropertyLens(p),v)
            end
        elseif comp isa DrivingField{T}
            for (p,v) in zip(params,range[i])
                new_df = set(new_df,PropertyLens(p),v)
            end
        elseif comp isa NumericalParameters{T}
            for (p,v) in zip(params,range[i])
                new_p  = set(new_p,PropertyLens(p),v)
            end
        end
        sweeplist[i] = Simulation(new_h,new_df,new_p,deepcopy(sim.observables),
                sim.unitscaling,sim.dimensions,name,
                joinpath(droplast(sim.datapath),ensname,name*"/"),
                joinpath(droplast(sim.plotpath),ensname,name*"/"))
    end


    return Ensemble(sweeplist,stringexpand_vector(params)*"_sweep_"*hashstring,
                joinpath(droplast(sim.datapath),ensname*"/"),
                joinpath(droplast(sim.plotpath),ensname*"/"))
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
