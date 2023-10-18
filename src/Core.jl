function runkxbatch!(
    sim::Simulation{T},
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    rhs_cc,
    rhs_cv;
    kwargs...) where {T<:Real}

    p   = getparams(sim)
    nkx = length(kxsamples)
    
    @inline function rhs!(du,u,p,t)
        @inbounds for i in 1:nkx
            du[i] = rhs_cc(t,u[i],u[i+nkx],kxsamples[i],ky)
        end
        
        @inbounds for i in nkx+1:2nkx
            du[i] = rhs_cv(t,u[i-nkx],u[i],kxsamples[i-nkx],ky)
        end
    end
    
    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    tspan          = (p.tsamples[1],p.tsamples[end])
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=p.tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)

    integrateobs_kxbatch_add!(sim,sol,kxsamples,ky,moving_bz)
end



"""
    run_simulation1d!(sim::Simulation{T};kwargs...)

Run a 1D simulation for a given Simulation `sim`.

This function should not be called by a user, since it has no convenience features
such as saving and plotting. Use [`run_simulation!`](@ref) instead

# Returns
The combined observables obtained from the simulation.

# See also
[`run_simulation!`](@ref), [`run_simulation2d!`](@ref)

"""
function run_simulation1d!(
    sim::Simulation{T},
    ky::T=zero(T);
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}

    p                   = getparams(sim)
    kxbatches           = padvecto_overlap!(subdivide_vector(p.kxsamples,kxbatch_basesize))
    γ1                  = oneunit(T) / p.t1
    γ2                  = oneunit(T) / p.t2    
    a                   = get_vecpotx(sim.drivingfield)
    f                   = get_efieldx(sim.drivingfield)
    ϵ                   = getϵ(sim.hamiltonian)
    dcc,dcv,dvc,dvv     = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))

    for kxs in kxbatches
        runkxbatch!(sim,kxs,ky,getmovingbz(sim,kxs),rhs_cc,rhs_cv;kwargs...)
    end

    return sim.observables    
end


"""
    run_simulation2d!(sim::Simulation{T};kwargs...)

Run a 2D simulation for a given Simulation `sim`.

This function should not be called by a user, since it has no convenience features
such as saving and plotting. Use [`run_simulation!`](@ref) instead

# Returns
The combined observables obtained from the simulation.

# See also
[`run_simulation!`](@ref), [`run_simulation1d!`](@ref)

"""
function run_simulation2d!(
    sim::Simulation{T};
    kyparallel=true,
    maxparallel_ky=64,
    kxbatch_basesize=512,
    threaded=true,
    kwargs...) where {T<:Real}

    if !kyparallel
        @info "Starting serial execution"
        return run_simulation2d!(sim;
            kxbatch_basesize=kxbatch_basesize,
            maxparallel_ky=1,
            threaded=threaded,
            kwargs...)
    end

    p                   = getparams(sim)
    kybatches           = padvecto_overlap!(subdivide_vector(p.kysamples,maxparallel_ky))
    observables_total   = deepcopy(sim.observables)

    if maxparallel_ky > 1
        @info "Starting parallel execution \n"*
        "Size of ky-batches: $maxparallel_ky"
    end

    @info "Size of kx-batches: $kxbatch_basesize"

    for kybatch in kybatches
        observables_buffer = run_kybatch!(sim,kybatch;
            kxbatch_basesize=kxbatch_basesize,
            threaded=threaded,
            kwargs...)

        for (o,otot) in zip(observables_buffer,observables_total)
            addto!(o,otot)
        end

        resize_obs!(sim)
        
        @everywhere GC.gc()
        @info "Batch finished"
    end

    sim.observables .= observables_total
    return sim.observables
end

function run_kybatch!(
    sim::Simulation{T},
    kysamples::AbstractVector{T};
    threaded=true,
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}

    if threaded
        obs = Folds.map(
            ky -> run_simulation1d!(
                deepcopy(sim),
                ky;
                kxbatch_basesize=kxbatch_basesize,
                kwargs...),
            kysamples)
        integrateobs_threaded!(obs,sim.observables,kysamples)
    else
        obs = pmap(
            ky -> run_simulation1d!(
                sim,
                ky;
                kxbatch_basesize=kxbatch_basesize,
                kwargs...),
            kysamples)
        integrateobs!(obs,sim.observables,kysamples)
    end


    return sim.observables
end

"""
    run_simulation!(sim::Simulation{T};
        savedata=true,
        saveplots=true,
        kxparallel=false,
        kwargs...)

Run a simulation.

# Arguments
- `sim::Simulation{T}`: The simulation object containing physical problem and results
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `nbatches::Int` : The number of batches being run in parallel (default is 
    4*max(nprocs,nthreads)).
- `kwargs...`: Additional keyword arguments.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d!`](@ref), [`run_simulation2d!`](@ref)

"""
function run_simulation!(
    sim::Simulation{T};
    savedata=true,
    saveplots=true,
    kyparallel=true,
    threaded=false,
    maxparallel_ky=64,
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)*
            "# threads: $(Threads.nthreads())\n"*
            "# processes: $(Distributed.nprocs())"


    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    if sim.dimensions==1
        run_simulation1d!(sim;kwargs...)
    else
        run_simulation2d!(sim;
            kyparallel=kyparallel,
            threaded=threaded,
            maxparallel_ky=maxparallel_ky,
            kxbatch_basesize=kxbatch_basesize,
            kwargs...)
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end
    if saveplots
        plotdata(sim;kwargs...)
        plotfield(sim)
    end

    return sim.observables
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
        allobs = collect(
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

            @everywhere GC.gc
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

