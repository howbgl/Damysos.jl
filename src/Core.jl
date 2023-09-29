export solve_pre!
function solve_pre!(prob,u0,tsamples,kx,ky,u;kwargs...)
    itgr = init(remake(prob,p=[kx,ky]),nothing;kwargs...)

    i = 1
    nt = length(tsamples)
    while i<=nt
        i=copysol!(i,itgr.t,tsamples,nt,itgr,u)
        step!(itgr)
    end 
end

export copysol!
function copysol!(i,t,ts,nt,itgr,u)
    while i<=nt && ts[i]<t
        #@info "copysol i:",i
        #@info size(itgr(ts[i]))
        u[:,i] .= itgr(ts[i])
        i+=1
    end
    return i
end

function solve_eom(rhs!,u0,tsamples,p::Vector{T};
    kwargs...) where {T<:Real}

    prob    = ODEProblem(rhs!,u0,(tsamples[1],tsamples[end]),p;)
    sol     = solve(prob;saveat=tsamples,kwargs...)
    return sol[:,:]
end

"""
        run_simulation1d!(sim::Simulation{T}, ky::T;
            savedata=true,
            saveplots=true,
            kxparallel=false,
            kwargs...)

    Run a 1D simulation for a given `sim` and wavenumber `ky`.

    # Arguments
    - `sim::Simulation{T}`: The simulation object.
    - `ky::T`: The wavenumber in the y-direction.
    - `savedata::Bool`: Whether to save data (default is `true`).
    - `saveplots::Bool`: Whether to save plots (default is `true`).
    - `kxparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
    - `kwargs...`: Additional keyword arguments.

    # Returns
    The observables obtained from the simulation.

    # See also
    [`run_simulation1d_serial!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref)

"""
function run_simulation1d!(sim::Simulation{T},rhs!,ky::T;
        savedata=true,
        saveplots=true,
        kwargs...) where {T<:Real}

    p              = getparams(sim)
    kx_samples     = p.kxsamples
    tsamples       = p.tsamples    
    a              = get_vecpotx(sim.drivingfield)
    ubuff          = zeros(Complex{T},2*p.nkx,p.nt)
    u0              = zeros(Complex{T},2)
    sim.observables .= [resize(o,sim.numericalparams) for o in sim.observables]
    moving_bz       = kx_samples .- a.(tsamples)' # nkx x nt matrix
    
    prob = ODEProblem(rhs!,u0,[tsamples[1],tsamples[end]],[0.1,ky])

    @floop for (i,kx) in enumerate(kx_samples)
        solve_pre!(prob,u0,tsamples,kx,ky,@view ubuff[2i-1:2i,:];abstol=p.atol,reltol=p.rtol)
        for o in sim.observables
            calc_obs_mode!(o,sim.hamiltonian,ubuff,kx .- a.(tsamples),ky,i)
        end
    end

    
    for o in sim.observables
        integrate1d_obs!(o,kx_samples,moving_bz)
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata == true
        Damysos.savedata(sim)
    end

    if saveplots == true
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
    [`run_simulation1d_serial!`](@ref), [`run_simulation1d!`](@ref), [`run_simulation!`](@ref)

"""
function run_simulation2d!(sim::Simulation{T};
                savedata=true,
                saveplots=true,
                kxparallel=false,
                nbatches=4*Threads.nthreads(),
                kwargs...) where {T<:Real}

    p              = getparams(sim)

    γ1              = oneunit(T) / p.t1
    γ2              = oneunit(T) / p.t2
    
    a              = get_vecpotx(sim.drivingfield)
    f              = get_efieldx(sim.drivingfield)
    ϵ              = getϵ(sim.hamiltonian)

    dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
            du[1] = rhs_cc(t,u[1],u[2],p[1],p[2])
            du[2] = rhs_cv(t,u[1],u[2],p[1],p[2])
    end
    last_obs    = run_simulation1d!(sim,rhs!,p.kysamples[1];
                    savedata=false,
                    saveplots=false,
                    kxparallel=kxparallel,
                    nbatches=nbatches,
                    kwargs...)
    total_obs   = zero.(deepcopy(last_obs))

    for i in 2:p.nky
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
        obs = run_simulation1d!(sim,rhs!,p.kysamples[i];
                savedata=false,
                saveplots=false,
                kxparallel=kxparallel,
                nbatches=nbatches,
                kwargs...)
        
        for (o,last,tot) in zip(obs,last_obs,total_obs)
            temp = integrate2d_obs([last,o],collect(p.kysamples[i-1:i]))
            addto!(temp,tot)
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
    [`run_simulation2d!`](@ref)

"""
function run_simulation!(sim::Simulation{T};
                    savedata=true,
                    saveplots=true,
                    nbatches=4*Threads.nthreads(),
                    kxparallel=false,
                    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    if sim.dimensions==1
        obs = run_simulation1d!(sim,zero(T);
                                savedata=savedata,
                                saveplots=saveplots,
                                nbatches=nbatches,
                                kwargs...)
    elseif sim.dimensions==2
        obs = run_simulation2d!(sim;
                                savedata=savedata,
                                saveplots=saveplots,
                                nbatches=nbatches,
                                kxparallel=kxparallel,
                                kwargs...)
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
    [`run_simulation2d!`](@ref)

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
        if kxparallel
            @warn "ensembleparallel & kxparallel = true\nCannot do both: just do ensemble"
        end
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
        push!(sims,Simulation(
                            sim.hamiltonian,
                            sim.drivingfield,
                            params,
                            deepcopy(sim.observables),
                            sim.unitscaling,
                            sim.dimensions,
                            sim.id,
                            sim.datapath,
                            sim.plotpath))
    end

    return sims
end
