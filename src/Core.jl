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
function run_simulation1d!(sim::Simulation{T},ky::T=zero(T);kwargs...) where {T<:Real}
    return run_simulation1d!(sim,ky,getmovingbz(sim);kwargs...)    
end

function run_simulation1d!(sim::Simulation{T},ky::T,moving_bz;kwargs...) where {T<:Real}

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
    sol            = solve(prob;saveat=p.tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)
    
    sim.observables .= calc_obs_k1d!(sim,sol,ky,moving_bz)
    
    return sim.observables  
end


function getrhs(sim::Simulation{T}) where {T<:Real}
    
    a,f,ϵ = let h=sim.hamiltonian,df=sim.drivingfield
        (get_vecpotx(df),get_efieldx(df),getϵ(h))
    end
    dcc,dcv,dvc,dvv = let h=sim.hamiltonian
        getdipoles_x(h)
    end
    γ1,γ2 = let p = getparams(sim)
        (1/p.t1,1/p.t2)
    end

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
            du[1] = rhs_cc(t,u[1],u[2],p[1],p[2])
            du[2] = rhs_cv(t,u[1],u[2],p[1],p[2])
    end

    return rhs!
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
function run_simulation2d!(sim::Simulation{T};
    kyparallel=true,
    maxparallel_ky=32,
    threaded=true,
    kwargs...) where {T<:Real}

    if !kyparallel
        @info "Starting serial execution"
        return run_simulation2d!(sim;maxparallel_ky=1,threaded=threaded,kwargs...)
    end

    p                   = getparams(sim)
    kybatches           = pad_kybatches!(subdivide_vector(p.kysamples,maxparallel_ky))

    if maxparallel_ky > 1
        @info "Starting parallel execution \nProcessing $maxparallel_ky ky-values simultaneously"
    end

    for kybatch in kybatches
        @info "Batch: $kybatch"
        observables_buffer = run_simulation2d_pbatch!(deepcopy(sim),kybatch;
            threaded=threaded,kwargs...)

        for (o,otot) in zip(observables_buffer,sim.observables)
            addto!(o,otot)
        end
        GC.gc()
        
        @info "Batch finished"
    end

    return sim.observables
end

function run_simulation2d_pbatch!(
    sim::Simulation{T},
    kysamples::AbstractVector{T};
    threaded=true,
    kwargs...) where {T<:Real}

    if threaded
        obs     = Folds.map(
            ky -> run_simulation1d!(deepcopy(sim),ky;kwargs...),
            kysamples)
    else
        obs     = pmap(
            ky -> run_simulation1d!(sim,ky;kwargs...),
            kysamples)
    end

    for (i,otot) in enumerate(sim.observables)
        integrate2d_obs!([o[i] for o in obs],otot,collect(kysamples))
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
function run_simulation!(sim::Simulation{T};
    savedata=true,
    saveplots=true,
    kyparallel=true,
    threaded=false,
    maxparallel_ky=32,
    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

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

