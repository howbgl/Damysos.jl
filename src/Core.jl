
function run_simulation1d!(sim::Simulation{T},ky::T;
        rtol=1e-10,atol=1e-10,savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p              = getparams(sim)
    
    γ              = 1.0 / p.t2
    nkx            = p.nkx
    kx_samples     = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    a              = get_vecpotx(sim.drivingfield)
    f              = get_efieldx(sim.drivingfield)
    ϵ              = getϵ(sim.hamiltonian)

    dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cv,kx,ky)     = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky))
    rhs_cv(t,cc,cv,kx,ky)  = (-γ - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        (2.0 * dvv(kx-a(t),ky) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
         for i in 1:nkx
              du[i] = rhs_cc(t,u[i+nkx],kx_samples[i],ky)
         end
    
         for i in nkx+1:2nkx
              du[i] = rhs_cv(t,u[i-nkx],u[i],kx_samples[i-nkx],ky)
         end
         return
    end

    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=tsamples,reltol=rtol,abstol=atol,kwargs...)
    
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
                savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p         = getparams(sim)
    total_obs = run_simulation1d!(sim,p.kysamples[1];
                    savedata=false,saveplots=false,kwargs...)
    last_obs  = deepcopy(total_obs)

    for i in 2:p.nky
        if mod(i,10)==0
            @info 100.0i/p.nky,"%"
        end
        obs = run_simulation1d!(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)
        
        for o in obs
            lasto = filter(x -> x isa typeof(o),last_obs)
            nexto = filter(x -> x isa typeof(o),obs)
            if length(lasto) != 1 || length(nexto) != 1
                @warn "length(lasto) != 1 || length(nexto) != 1"
            end

            integrate2d_obs!(sim,[lasto[1],nexto[1]],collect(p.kysamples[i-1:i]),total_obs) 
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

function run_simulation!(sim::Simulation{T};kwargs...) where {T<:Real}

    @info "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    if sim.dimensions==1
        obs = run_simulation1d!(sim,0.0;kwargs...)
    elseif sim.dimensions==2
        obs = run_simulation2d!(sim;kwargs...)
    end
    savemetadata(sim)
    return obs
end

function run_simulation!(ens::Ensemble{T};savedata=true,saveplots=true,
                ensembleparallel=true,
                makecombined_plots=true,
                kwargs...) where {T<:Real}

    allobs = []

    if ensembleparallel
        allobs = Folds.collect(
            run_simulation!(s;savedata=savedata,saveplots=saveplots,kwargs...)
            for s in ens.simlist)
    else
        for i in eachindex(ens.simlist)
            obs = run_simulation!(ens.simlist[i];savedata=savedata,saveplots=saveplots,kwargs...)
            push!(allobs,obs)
        end
    end

    if makecombined_plots == true
        Damysos.plotdata(ens)
    end

    savemetadata(ens)

    return allobs
end
