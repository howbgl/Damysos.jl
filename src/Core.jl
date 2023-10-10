
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
        ky::T;
        kwargs...) where {T<:Real}

    sol = zeros(Complex{T},2*length(kxs),length(ts))

    @inbounds for i in eachindex(kxs)
        solve_pre!(eomintegrator,ts,kxs[i],ky,@view sol[2i-1:2i,:];kwargs...)
    end

    GC.gc()
    return sol
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

function run_simulation_kxline!(
    eomintegrator,
    kxsamples::AbstractVector{T},
    ky::T,
    tsamples::AbstractVector{T},
    moving_bz::Matrix{T},
    observables::Vector{Observable{T}},
    observable_funcs;    
    kwargs...) where {T<:Real}

    observables_buffer  = deepcopy(observables) # contains result for last kx for trapz 
    sol                 = zeros(Complex{T},2,length(tsamples))
    for (i,kx) in enumerate(kxsamples)
        solve_pre!(eomintegrator,tsamples,kx,ky,sol;kwargs...)
        for (o,obuff,funcs) in zip(observables,observables_buffer,observable_funcs)
            calc_obs_mode!(o,sol,tsamples,kx,ky,funcs)
            if i>1
                integrate_obs!([obuff,o],o,[kx,kxsamples[i-1]])
            end
            copyto!(obuff,o)
        end
    end
    
    return observables  
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
    threaded=false,
    maxparallel_ky=64,
    kwargs...) where {T<:Real}

    p              = getparams(sim)
    kxsamples      = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    u0          = zeros(Complex{T},2)
    rhs!        = getrhs(sim)
    prob        = ODEProblem(rhs!,u0,tspan,zeros(T,2))
    itgr        = init(prob;abstol=p.atol,reltol=p.rtol)
    moving_bz   = get_movingbz(sim.drivingfield,p)

    sim.observables     .= [resize(o,sim.numericalparams) for o in sim.observables]
    kybatches           = pad_kybatches(subdivide_vector(p.kysamples,maxparallel_ky))
    observable_funcs    = [get_funcs(o,sim) for o in sim.observables]

    @info "Using $(Distributed.nprocs()) workers for $maxparallel_ky batches"

    for batch in kybatches
        observables_buffer = pmap(
            ky -> run_simulation_kxline!(
                itgr,
                kxsamples,
                ky,
                tsamples,
                moving_bz,
                sim.observables,
                observable_funcs),
            batch)
        for (i,o) in enumerate(sim.observables)
            integrate_obs_add!([ob[i] for ob in observables_buffer],o,batch)
        end
        @everywhere GC.gc()
        @info "Batch finished"
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
    maxparallel_ky=64,
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

