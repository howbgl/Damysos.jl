
function run_simulation1d(sim::Simulation{T},ky::T;rtol=1e-10,atol=1e-10,savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p              = getparams(sim)
    
    γ              = 1.0 / p.t2
    nkx            = p.nkx
    kx_samples     = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    a              = get_vecpot(sim.drivingfield)
    f              = get_efield(sim.drivingfield)
    ϵ              = getϵ(sim.hamiltonian)

    dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cv,kx,ky)     = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky))
    rhs_cv(t,cc,cv,kx,ky)  = (-γ - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * (2.0 * dvv(kx-a(t),ky) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


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
    
    obs = calc_obs(sim,sol)

    if savedata == true
        Damysos.savedata(sim,obs)
    end

    if saveplots == true
        Damysos.plotdata(sim)
    end

    return obs
end

function run_simulation2d(sim::Simulation{T};savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p         = getparams(sim)
    total_obs = []
    last_obs  = run_simulation1d(sim,p.kysamples[1];savedata=false,saveplots=false,kwargs...)

    for i in 2:p.nky
        if mod(i,10)==0
            println(100.0i/p.nky,"%")
        end
        obs = run_simulation1d(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)
        for o in obs
            push!(total_obs,trapz((:,hcat(p.kysamples[i-1],p.kysamples[i])),hcat(last_obs[j],o)))
        end
        last_obs = deepcopy(obs)
    end

    if savedata == true
        Damysos.savedata(sim,total_obs)
    end

    if saveplots == true
        Damysos.plotdata(sim,datapath)
    end

    return total_obs
end

function run_simulation(sim::Simulation{T};kwargs...) where {T<:Real}
    if sim.dimensions==1
        obs = run_simulation1d(sim,0.0;kwargs...)
    elseif sim.dimensions==2
        obs = run_simulation2d(sim;kwargs...)
    end
    saveparams(sim)
    return obs
end

function run_simulation(ens::Ensemble{T};savedata=true,saveplots=true,
                makecombined_plots=true,kwargs...) where {T<:Real}

    allobs = []

    for i in eachindex(ens.simlist)
        obs = run_simulation(ens.simlist[i];savedata=savedata,saveplots=saveplots,kwargs...)
        push!(allobs,obs)
    end

    if makecombined_plots == true
        Damysos.plotdata(ens,allobs)
    end

    return allobs
end
