
export solve_pre!
function solve_pre!(eomintegrator,tsamples,kx,ky,sol;kwargs...)

    reinit!(eomintegrator)
    eomintegrator.p = [kx,ky]

    i = 1
    nt = length(tsamples)
    while i<=nt
        i=copysol!(i,eomintegrator.t,tsamples,nt,eomintegrator,sol)
        step!(eomintegrator)
    end 
end

export copysol!
function copysol!(i,t,ts,nt,itgr,sol)
    @inbounds while i<=nt && ts[i]<t
        @inbounds sol[:,i] .= itgr(ts[i])
        i+=1
    end
    return i
end

export solve_eom!
function solve_eom!(
        eomintegrator,
        ts::AbstractVector{T},
        kxs::AbstractVector{T},
        ky::T,
        sol;
        kwargs...) where {T<:Real}

    @inbounds for i in eachindex(kxs)
        solve_pre!(eomintegrator,ts,kxs[i],ky,@view sol[2i-1:2i,:];kwargs...)
    end
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
function run_simulation1d!(sim::Simulation{T};kwargs...) where {T<:Real}
    @warn "1d not implemented yet!"
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
function run_simulation2d!(sim::Simulation{T};kwargs...) where {T<:Real}

    p              = getparams(sim)
    kxs            = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    
    u0          = zeros(Complex{T},2)
    rhs!        = getrhs(sim)
    prob        = ODEProblem(rhs!,u0,tspan,zeros(T,2))
    itgr        = init(prob;abstol=p.atol,reltol=p.rtol)
    sol         = zeros(Complex{T},2*p.nkx,p.nt)
    moving_bz   = get_movingbz(sim.drivingfield,p)

    sim.observables     .= [resize(o,sim.numericalparams) for o in sim.observables]
    last_obs            = deepcopy(sim.observables)
    next_obs            = deepcopy(last_obs)
    observable_funcs    = [get_funcs(o,sim) for o in sim.observables]
    
    solve_eom!(itgr,tsamples,kxs,p.kysamples[1],sol;abstol=p.atol,reltol=p.rtol)
    calc_allobs_1d!(sim,last_obs,sol,p,p.kysamples[1],moving_bz,observable_funcs)

    @inbounds for i in 2:p.nky
        solve_eom!(itgr,tsamples,kxs,p.kysamples[i],sol;
                        abstol=p.atol,reltol=p.rtol)
        calc_allobs_1d!(sim,next_obs,sol,p,p.kysamples[i],moving_bz,observable_funcs)

        for (o,last,tot) in zip(next_obs,last_obs,sim.observables)
            integrate2d_obs_add!([last,o],tot,collect(p.kysamples[i-1:i]))
            Damysos.copyto!(last,o)
        end
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)
    @debug "Batch completed."
    GC.gc()

    return sim.observables
end

function run_simulation_parallel!(sim::Simulation{T};
    threaded=false,
    nbatches=4*Distributed.nprocs(),
    kwargs...) where {T<:Real}

    sims        = makekxbatches(sim,nbatches)
    obs_batches = Vector{Vector{Observable{T}}}(undef,nbatches)

    if threaded
        @info "using $nbatches batches for $(Threads.nthreads()) workers (threaded)"
        if sim.dimensions==1
            obs_batches = Folds.map(s -> run_simulation1d!(s;savedata=false,
                                                saveplots=false,
                                                kwargs...),sims)
        else
            obs_batches = Folds.map(s -> run_simulation2d!(s;kwargs...),sims)
        end
    else
        @info "using $nbatches batches for $(Distributed.nprocs()-1) workers (distributed)"
        if sim.dimensions==1
            obs_batches = @showprogress pmap(s -> run_simulation1d!(s;savedata=false,
                                                saveplots=false,
                                                kwargs...),sims)
        else
            obs_batches = @showprogress pmap(s -> run_simulation2d!(s;kwargs...),sims)
        end
    end

    @info "All batches finished, summing up..."
    for obs in obs_batches
        for (o,otot) in zip(obs,sim.observables)
            addto!(otot,o)
        end
    end
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
    kxparallel=true,
    threaded=false,
    nbatches=4*(Distributed.nprocs()>Threads.nthreads() ? Distributed.nprocs() : Threads.nthreads()),
    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    if kxparallel
        run_simulation_parallel!(sim;nbatches=nbatches,threaded=threaded)
    else
        if sim.dimensions==1
            run_simulation1d!(sim;kwargs...)
        else
            run_simulation2d!(sim;kwargs...)
        end
    end

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

"""
        makekxbatches(sim::Simulation{T}, nbatches::U) where {T<:Real, U<:Integer}

    Divide a simulation into multiple batches along the kx direction.

    This function is automatically called by [`run_simulation!`](@ref). 
    It takes a simulation and divides it into `nbatches` simulations, 
    each covering a portion of the kx range. It is useful for parallelizing simulations 
    along the kx axis.

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
        params = NumericalParams2dSlice(deepcopy(sim.numericalparams),(allkxs[lidx],allkxs[ridx]))
        s      = Simulation(
                            deepcopy(sim.hamiltonian),
                            deepcopy(sim.drivingfield),
                            deepcopy(params),
                            convert(Vector{Observable{T}},empty.(sim.observables)),
                            deepcopy(sim.unitscaling),
                            deepcopy(sim.dimensions),
                            deepcopy(sim.id),
                            deepcopy(sim.datapath),
                            deepcopy(sim.plotpath))
        push!(sims,s)
    end

    return sims
end
