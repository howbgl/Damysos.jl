
export solve_pre!
function solve_pre!(prob,tsamples,kx,ky,u;kwargs...)

    itgr = init(remake(prob;p=[kx,ky]);kwargs...)

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

function solve_eom(
        prob::ODEProblem{Vector{Complex{T}}},
        ts::AbstractVector{T},
        kxs::AbstractVector{T},
        ky::T;
        kwargs...) where {T<:Real}


    sol = zeros(Complex{T},2*length(kxs),length(ts))
    
    for i in eachindex(kxs)
        solve_pre!(prob,ts,kxs[i],ky,@view sol[2i-1:2i,:];kwargs...)
    end

    return sol
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
    [`run_simulation1d_serial!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref), 
    [`run_simulation!(ens::Ensemble{T})`](@ref)

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
    [`run_simulation1d!`](@ref), [`run_simulation!`](@ref), 
    [`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation2d!(sim::Simulation{T};kwargs...) where {T<:Real}

    p              = getparams(sim)
    kxs            = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    sim.observables .= [resize(o,sim.numericalparams) for o in sim.observables]
 
    u0          = zeros(Complex{T},2)
    rhs!        = getrhs(sim)
    prob        = ODEProblem(rhs!,u0,tspan,[0.0,0.0])
    sol         = solve_eom(prob,tsamples,kxs,p.kysamples[1];abstol=p.atol,reltol=p.rtol)

    last_obs    = calc_obs_k1d(sim,sol,p.kysamples[1])
    total_obs   = zero.(deepcopy(sim.observables))

    for i in 2:p.nky
        
        sol = solve_eom(prob,tsamples,kxs,p.kysamples[1];abstol=p.atol,reltol=p.rtol)
        obs = calc_obs_k1d(sim,sol,p.kysamples[i])

        for (o,last,tot) in zip(obs,last_obs,total_obs)
            temp = integrate2d_obs([last,o],collect(p.kysamples[i-1:i]))
            addto!(temp,tot)
        end
        last_obs = deepcopy(obs)

        if mod(i,10)==0
            GC.gc()
        end
    end

    sim.observables .= total_obs
    normalize!.(sim.observables,(2π)^sim.dimensions)
    @info "Batch completed."

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
                    nbatches=4*Distributed.nprocs(),
                    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    @info "pmap"
    sims        = makekxbatches(sim,nbatches)
    @info "using $nbatches batches for $(Distributed.nprocs()) workers"
    res = Vector{Vector{Observable{T}}}(undef,length(sims))
    if sim.dimensions==1
        res = pmap(s -> run_simulation1d!(s;savedata=false,
                                            saveplots=false,
                                            kwargs...),sims)
    else
        res = pmap(s -> run_simulation2d!(s;kwargs...),sims)
    end
    @info "All batches finished, summing up..."
    
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
