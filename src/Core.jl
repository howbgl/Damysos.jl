
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
    while i<=nt && ts[i]<t
        # @info "copysol i:",i
        #@info size(itgr(ts[i]))
        sol[:,i] .= itgr(ts[i])
        i+=1
    end
    return i
end

function solve_eom!(
        eomintegrator,
        ts::AbstractVector{T},
        kxs::AbstractVector{T},
        ky::T,
        sol;
        kwargs...) where {T<:Real}

    for i in eachindex(kxs)
        solve_pre!(eomintegrator,ts,kxs[i],ky,@view sol[2i-1:2i,:];kwargs...)
    end
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
    
    
    u0          = zeros(Complex{T},2)
    rhs!        = getrhs(sim)
    prob        = ODEProblem(rhs!,u0,tspan,[0.0,0.0])
    itgr        = init(prob;abstol=p.atol,reltol=p.rtol)
    sol         = zeros(Complex{T},2*p.nkx,p.nt)
    moving_bz   = get_movingbz(sim.drivingfield,p)

    sim.observables     .= [resize(o,sim.numericalparams) for o in sim.observables]
    last_obs            = deepcopy(sim.observables)
    next_obs            = deepcopy(last_obs)
    buff_obs            = deepcopy(next_obs)
    
    solve_eom!(itgr,tsamples,kxs,p.kysamples[1],sol;abstol=p.atol,reltol=p.rtol)

    calc_allobs_1d!(sim,last_obs,sol,p,p.kysamples[1],moving_bz)

    for i in 2:p.nky
        solve_eom!(itgr,tsamples,kxs,p.kysamples[i],sol;
                        abstol=p.atol,reltol=p.rtol)
        calc_allobs_1d!(sim,next_obs,sol,p,p.kysamples[i],moving_bz)

        for (o,last,buff,tot) in zip(next_obs,last_obs,buff_obs,sim.observables)
            integrate2d_obs!([last,o],buff,collect(p.kysamples[i-1:i]))
            addto!(tot,buff)
            Damysos.copyto!(last,o)
        end

        # if mod(i,10)==0
        #     GC.gc()
        # end
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)
    @info "Batch completed."

    return sim.observables
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
        for (obs,tot_obs) in zip(r,total_res)
            addto!(tot_obs,obs)
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
