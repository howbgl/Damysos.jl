"""
solve_eom(sim::Simulation{T}, ky::T,ky_index::Integer;kwargs...)

Solve EOM of a 1D slice simulation for a given `sim` and wavenumber `ky`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `ky::T`: The wavenumber in the y-direction.
- `ky_index::Integer`: The index of the wavenumber in the y-direction.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function solve_eom(sim::Simulation{T},ky::T,ky_index::Integer;
    kwargs...) where {T<:Real}

    p              = getparams(sim)

    γ1              = oneunit(T) / p.t1
    γ2              = oneunit(T) / p.t2

    nkx            = p.nkx
    kx_samples     = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    a              = get_vecpotx(sim.drivingfield)
    f              = get_efieldx(sim.drivingfield)
    ϵ              = getϵ(sim.hamiltonian)

    dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
        @inbounds for i in 1:nkx
            du[i] = rhs_cc(t,u[i],u[i+nkx],kx_samples[i],ky)
        end
    
        @inbounds for i in nkx+1:2nkx
            du[i] = rhs_cv(t,u[i-nkx],u[i],kx_samples[i-nkx],ky)
        end
    end

    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)
    
    return sol
end

"""
run_simulation1d!(sim::Simulation{T}, ky::T,ky_index::Integer;
        savedata=true,
        saveplots=true,
        kxparallel=false,
        kwargs...)

Run a 1D simulation for a given `sim` and wavenumber `ky`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `ky::T`: The wavenumber in the y-direction.
- `ky_index::Integer`: The index of the wavenumber in the y-direction.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d_serial!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation1d!(sim::Simulation{T},ky::T,ky_index::Integer;
        savedata=true,
        saveplots=true,
        kwargs...) where {T<:Real}

    p              = getparams(sim)
    
    sol = solve_eom(sim,ky,ky_index;kwargs...)
    
    
    calc_obs_k1d!(sim,sol,ky,ky_index)
    finalize_obs!(sim)
    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end

    if saveplots
        Damysos.plotdata(sim)
        plotfield(sim)
    end

    return sim.observables
end



"""
    run_simulation2d!(sim::Simulation{T};
        savedata=true,
        saveplots=true,
        kxparallel=false,
        kwargs...)

Run a 2D simulation for a given `sim`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# Returns
The combined observables obtained from the simulation.

# See also
[`run_simulation1d_serial!`](@ref), [`run_simulation1d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation2d!(sim::Simulation{T};
                savedata=true,
                saveplots=true,
                kyparallel=false,
                kwargs...) where {T<:Real}

    p = getparams(sim)
    
    for i in 1:p.nky
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
        run_simulation1d!(sim,p.kysamples[i],i;savedata=false,saveplots=false,kwargs...)
    end

    integrate2d_obs!(sim)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end

    if saveplots
        Damysos.plotdata(sim)
        plotfield(sim)
    end

    return total_obs

end


"""
    run_simulation!(sim::Simulation{T};
        savedata=true,
        saveplots=true,
        kxparallel=false,
        kwargs...)

Run a simulation for a given `sim`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d_serial!`](@ref), [`run_simulation1d!`](@ref), 
[`run_simulation2d!`](@ref), [`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation!(sim::Simulation{T};
                    savedata=true,
                    saveplots=true,
                    kxparallel=false,
                    kx_workers=128,
                    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    @info "floop"
    if kxparallel
        sims        = makekxbatches(sim,minimum([kx_workers,Threads.nthreads()]))
        @info "using $(length(sims)) tasks"
        res = Vector{Vector{Observable{T}}}(undef,length(sims))
        if sim.dimensions==1
              res = Folds.collect(solve_eom(s,zero(T);
                                                    savedata=false,
                                                    saveplots=false,kwargs...) for s in sims)
        else
            res = Folds.collect(run_simulation2d!(s;savedata=false,
                                                    saveplots=false,
                                                    kxparallel=false,kwargs...) for s in sims)
        end
        
        total_res   = deepcopy(res[1])
        popfirst!(res) # do not add first entry twice!
        for r in res
            for (i,obs) in enumerate(r)
                addto!(obs,total_res[i])
            end
        end
        sim.observables .= total_res

        if savedata
            Damysos.savedata(sim)
        end
        if saveplots
            plotdata(sim;kwargs...)
            plotfield(sim)
        end

        return total_res
    else
        if sim.dimensions==1
            return solve_eom(sim,zero(T);
                                            savedata=false,
                                            saveplots=false,kwargs...)
        else
            return run_simulation2d!(sim;
                                    savedata=false,
                                    saveplots=false,
                                    kxparallel=false,kwargs...)
        end
    end


    if savedata
        savemetadata(sim)
    end
    
    return obs
end


"""
    run_simulation!(ens::Ensemble{T};
        savedata=true,
        saveplots=true,
        ensembleparallel=false,
        kxparallel=false,
        makecombined_plots=true,
        kwargs...)

Run simulations for an ensemble of `sim` objects.

# Arguments
- `ens::Ensemble{T}`: The ensemble of simulation objects.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `ensembleparallel::Bool`: Whether to run ensemble simulations in parallel (default is `false`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `makecombined_plots::Bool`: Whether to make combined plots (default is `true`).
- `kwargs...`: Additional keyword arguments.

# Returns
An array of observables obtained from the simulations.

# See also
[`run_simulation1d_serial!`](@ref), [`run_simulation1d!`](@ref), 
[`run_simulation2d!`](@ref), [`run_simulation!`](@ref)

"""
function run_simulation!(ens::Ensemble{T};
                savedata=true,
                saveplots=true,
                ensembleparallel=false,
                kxparallel=false,
                makecombined_plots=true,
                kwargs...) where {T<:Real}

    ensurepath(ens.datapath)
    ensurepath(ens.plotpath)

    allobs = []

    if ensembleparallel
        allobs = Folds.collect(
            run_simulation!(s;savedata=savedata,saveplots=false,kxparallel=false,kwargs...)
            for s in ens.simlist)
        # Run plotting sequentially
        if saveplots
            Damysos.plotdata.(ens.simlist;kwargs...)
        end
        
    else
        for i in eachindex(ens.simlist)
            obs = run_simulation!(ens.simlist[i];
                        savedata=savedata,
                        saveplots=saveplots,
                        kxparallel=kxparallel,
                        kwargs...)
            push!(allobs,obs)
        end
    end

    if makecombined_plots == true
        Damysos.plotdata(ens)
    end

    if savedata
        savemetadata(ens)
    end

    return allobs
end

"""
    makekxbatches(sim::Simulation{T}, nbatches::U) where {T<:Real, U<:Integer}

Divide a 1D simulation into multiple batches along the kx direction.

This function takes a 1D simulation and divides it into `nbatches` simulations, each covering a portion of the kx range. It is useful for parallelizing simulations along the kx axis.

# Arguments
- `sim::Simulation{T}`: The original 1D simulation object.
- `nbatches::U`: The number of batches to divide the simulation into.

# Returns
An array of simulation objects representing the divided batches.

# See also
[`run_simulation!`](@ref), [`run_simulation1d!`](@ref), [`run_simulation2d!`](@ref)

"""
function makekxbatches(sim::Simulation{T},nbatches::U) where {T<:Real,U<:Integer}
    
    p           = getparams(sim)
    if nbatches > p.nkx/2
        nbatches = floor(U,p.nkx/2)
    end

    allkxs      = p.kxsamples
    nper_batch  = fld(p.nkx,nbatches)
    sims        = empty([sim])

    for i in 1:nbatches
        lidx = max(1,(i-1)*nper_batch)
        if i==nbatches # In the last batch include all the rest
            ridx = length(allkxs)
        else
            ridx = i*nper_batch
        end
        params = NumericalParams2dSlice(sim.numericalparams,(allkxs[lidx],allkxs[ridx]))
        s      = Simulation(
                            sim.hamiltonian,
                            sim.drivingfield,
                            params,
                            deepcopy(sim.observables),
                            sim.unitscaling,
                            sim.dimensions,
                            sim.id,
                            sim.datapath,
                            sim.plotpath)
        resize_obs!(s)
        push!(sims,s)
    end

    return sims
end

export run_simulation2d_new!
function run_simulation2d_new!(sim::Simulation{T};
    savedata=true,
    saveplots=true,
    kxparallel=false,
    kwargs...) where {T<:Real}

    p         = getparams(sim)
    total_obs = deepcopy(solve_eom(sim,p.kysamples[1];
            savedata=false,saveplots=false,kwargs...))
    last_obs  = deepcopy(total_obs)

    for i in 2:p.nky
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
        obs = solve_eom(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)

        for (tot,o) in zip(total_obs,obs)
            addto!(o,tot)
        end
        last_obs = deepcopy(obs)
    end

    sim.observables .= total_obs

    if savedata == true
        Damysos.savedata(sim)
    end

    if saveplots == true
        Damysos.plotdata(sim)
        plotfield(sim)
    end

    return total_obs
end
