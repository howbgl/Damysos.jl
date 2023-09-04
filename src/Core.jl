"""
solve_eom(sim::Simulation{T}, ky::T,ky_index::Integer;kwargs...)

Solve EOM of a 1D slice simulation for a given `sim` and wavenumber `ky`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `ky::T`: The wavenumber in the y-direction.
- `ky_index::Integer`: The index of the wavenumber in the y-direction.
- `kwargs...`: Additional keyword arguments are passed to solve of DifferentialEquations.jl.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function solve_eom(sim::Simulation{T},ky::T,ky_index::Integer;
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
    sol            = solve(prob;saveat=tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)
    
    return sol
end

export run_simulation1d!
"""
run_simulation1d!(sim::Simulation{T}, ky::T,ky_index::Integer;
        savedata=true,
        saveplots=true,
        kyparallel=false,
        kwargs...)

Run a 1D simulation for a given `sim` and wavenumber `ky`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `ky::T`: The wavenumber in the y-direction.
- `ky_index::Integer`: The index of the wavenumber in the y-direction.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kyparallel::Bool`: Whether to run ky-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# See also
[`run_simulation1d_serial!`](@ref), [`run_simulation2d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation1d!(sim::Simulation{T},ky::T,ky_index::Integer;kwargs...) where {T<:Real}

    sol = solve_eom(sim,ky,ky_index;kwargs...)    
    calc_obs_k1d!(sim,sol,ky,ky_index)
    finalize_obs1d!(sim)
end

export run_simulation!
"""
    run_simulation!(sim::Simulation{T};
        savedata=true,
        saveplots=true,
        kyparallel=false,
        kwargs...)

Run a simulation for a given `sim`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kyparallel::Bool`: Whether to run ky-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d!`](@ref),[`run_simulation2d!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation!(sim::Simulation{T};
                    savedata=true,
                    saveplots=true,
                    kyparallel=false,
                    kwargs...) where {T<:Real}
    
    @info   "$(now())\nOn $(gethostname()):\n"*
            "Starting $(getshortname(sim)) (id: $(sim.id))\n"*printparamsSI(sim)

    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    init_obs!(sim)
    zero.(sim.observables)

    if sim.dimensions==1
        run_simulation1d!(sim,zero(T),1;kwargs...)
    else
        run_simulation2d!(sim;kyparallel=kyparallel,kwargs...)
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end

    if saveplots
        Damysos.plotdata(sim)
        plotfield(sim)
    end

    return sim.observables
end


"""
    run_simulation!(ens::Ensemble{T};
        savedata=true,
        saveplots=true,
        ensembleparallel=false,
        kyparallel=false,
        makecombined_plots=true,
        kwargs...)

Run simulations for an ensemble of `sim` objects.

# Arguments
- `ens::Ensemble{T}`: The ensemble of simulation objects.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `ensembleparallel::Bool`: Whether to run ensemble simulations in parallel (default is `false`).
- `kyparallel::Bool`: Whether to run ky-parallel simulations (default is `false`).
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
                kyparallel=false,
                makecombined_plots=true,
                kwargs...) where {T<:Real}

    ensurepath(ens.datapath)
    ensurepath(ens.plotpath)

    allobs = []

    if ensembleparallel
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

export run_simulation2d!
"""
    run_simulation2d!(sim::Simulation{T};
        savedata=true,
        saveplots=true,
        kyparallel=false,
        kwargs...)

Run a 2D simulation for a given `sim`.

# Arguments
- `sim::Simulation{T}`: The simulation object.
- `savedata::Bool`: Whether to save data (default is `true`).
- `saveplots::Bool`: Whether to save plots (default is `true`).
- `kyparallel::Bool`: Whether to run kx-parallel simulations (default is `false`).
- `kwargs...`: Additional keyword arguments.

# See also
[`run_simulation1d!`](@ref), [`run_simulation!`](@ref), 
[`run_simulation!(ens::Ensemble{T})`](@ref)

"""
function run_simulation2d!(sim::Simulation{T};
    kyparallel=false,
    kwargs...) where {T<:Real}

    p = getparams(sim)

    for (i,ky) in enumerate(p.kysamples)
        sol = solve_eom(sim,ky,i;kwargs...)
        calc_obs_k1d!(sim,sol,ky,i)
        if mod(i,2)==0
            @info "$(100.0i/p.nky)%"
        end
    end

    integrate2d_obs!(sim)
end
