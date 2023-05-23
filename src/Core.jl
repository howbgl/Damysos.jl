
function run_simulation1d!(sim::Simulation{T},ky::T;
        savedata=true,
        saveplots=true,
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

    rhs_cc(t,cv,kx,ky)     = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) - γ1
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        (2.0 * dvv(kx-a(t),ky) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
        @inbounds for i in 1:nkx
            du[i] = rhs_cc(t,u[i+nkx],kx_samples[i],ky)
        end
    
        @inbounds for i in nkx+1:2nkx
            du[i] = rhs_cv(t,u[i-nkx],u[i],kx_samples[i-nkx],ky)
        end
    end

    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)
    
    sim.observables .= calc_obs_k1d(sim,sol,ky)

    if savedata == true
        Damysos.savedata(sim)
    end

    if saveplots == true
        Damysos.plotdata(sim)
        plotfield(sim)
    end

    return sim.observables
end

function run_simulation2d!(sim::Simulation{T};
                savedata=true,
                saveplots=true,
                kyparallel=false,
                kwargs...) where {T<:Real}

    if kyparallel
        @info "Parallelizing over ky"
        sims        = makekybatches(sim,Threads.nthreads())
        res         = Folds.collect(run_simulation2d!(s;
                                            savedata=false,
                                            saveplots=false,
                                            kyparallel=false,kwargs...) for s in sims)
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
    end
    
    p           = getparams(sim)
    last_obs    = deepcopy(run_simulation1d!(sim,p.kysamples[1];
                    savedata=false,saveplots=false,kwargs...))
    total_obs   = zero.(last_obs)

    for i in 2:p.nky
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
        obs = run_simulation1d!(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)
        
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

function run_simulation!(sim::Simulation{T};
                    savedata=true,
                    saveplots=true,
                    kyparallel=false,
                    kwargs...) where {T<:Real}

    @info "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)
    
    if sim.dimensions==1
        obs = run_simulation1d!(sim,zero(T);savedata=savedata,saveplots=saveplots,kwargs...)
    elseif sim.dimensions==2
        obs = run_simulation2d!(sim;savedata=savedata,saveplots=saveplots,
                                kyparallel=kyparallel,kwargs...)
    end

    if savedata
        savemetadata(sim)
    end
    
    return obs
end

function run_simulation!(ens::Ensemble{T};
                savedata=true,
                saveplots=true,
                ensembleparallel=false,
                kyparallel=false,
                makecombined_plots=true,
                kwargs...) where {T<:Real}

    allobs = []

    if ensembleparallel
        if kyparallel
            @warn "ensembleparallel & kyparallel = true\nCannot do both: just do ensemble"
        end
        allobs = Folds.collect(
            run_simulation!(s;savedata=savedata,saveplots=false,kyparallel=false,kwargs...)
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
                        kyparallel=kyparallel,
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


function makekybatches(sim::Simulation{T},nbatches::U) where {T<:Real,U<:Integer}
    
    p           = getparams(sim)
    if nbatches > p.nky/2
        nbatches = floor(U,p.nky/2)
    end

    allkys      = p.kysamples
    nper_batch  = fld(p.nky,nbatches)
    sims        = empty([sim])

    for i in 1:nbatches
        lidx = max(1,(i-1)*nper_batch)
        if i==nbatches # In the last batch include all the rest
            ridx = length(allkys)
        else
            ridx = i*nper_batch
        end
        params = NumericalParams2dSlice(sim.numericalparams,(allkys[lidx],allkys[ridx]))
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

export run_simulation2d_add!
function run_simulation2d_add!(sim::Simulation{T};
    savedata=true,
    saveplots=true,
    kyparallel=false,
    kwargs...) where {T<:Real}

    if kyparallel
        @info "Parallelizing over ky"
        sims        = makekybatches(sim,Threads.nthreads())
        res         = Folds.collect(run_simulation2d!(s;
                                            savedata=false,
                                            saveplots=false,
                                            kyparallel=false,kwargs...) for s in sims)
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
            plotdata(sim,kwargs...)
            plotfield(sim)
        end

        return total_res
    end

    p         = getparams(sim)
    total_obs = deepcopy(run_simulation1d!(sim,p.kysamples[1];
            savedata=false,saveplots=false,kwargs...))
    last_obs  = deepcopy(total_obs)

    for i in 2:p.nky
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
        obs = run_simulation1d!(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)

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
