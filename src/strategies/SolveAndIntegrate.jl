function runkxbatch!(
    sim::Simulation{T},
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    rhs_cc,
    rhs_cv,
    obsfuncs;
    kwargs...) where {T<:Real}

    p           = getparams(sim)
    nkx         = length(kxsamples)
    
    @inline function rhs!(du,u,p,t)
        @inbounds for i in 1:nkx
            du[i] = rhs_cc(t,u[i],u[i+nkx],kxsamples[i],ky)
        end
        
        @inbounds for i in nkx+1:2nkx
            du[i] = rhs_cv(t,u[i-nkx],u[i],kxsamples[i-nkx],ky)
        end
    end
    
    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    tspan          = (p.tsamples[1],p.tsamples[end])
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=p.tsamples,reltol=p.rtol,abstol=p.atol,kwargs...)

    integrateobs_kxbatch_add!(sim,sol,kxsamples,ky,moving_bz,obsfuncs)
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
function run_simulation1d!(
    sim::Simulation{T},
    ky::T=zero(T);
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}

    p                   = getparams(sim)
    obsfuncs            = [getfuncs(sim,o) for o in sim.observables]
    kxbatches           = padvecto_overlap!(subdivide_vector(p.kxsamples,kxbatch_basesize))
    γ1                  = oneunit(T) / p.t1
    γ2                  = oneunit(T) / p.t2    
    a                   = get_vecpotx(sim.drivingfield)
    f                   = get_efieldx(sim.drivingfield)
    ϵ                   = getϵ(sim.hamiltonian)
    dcc,dcv,dvc,dvv     = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))

    for kxs in kxbatches
        runkxbatch!(sim,kxs,ky,getmovingbz(sim,kxs),rhs_cc,rhs_cv,obsfuncs;kwargs...)
        GC.gc()
    end

    return sim.observables    
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
function run_simulation2d!(sim::Simulation;
    kxbatch_basesize=128,
    maxparallel_ky=64,
    threaded=false,
    kwargs...)

    p                   = getparams(sim)
    kybatches           = padvecto_overlap!(subdivide_vector(p.kysamples,maxparallel_ky))
    
    return run_simulation2d!(sim,kybatches;
        kxbatch_basesize=kxbatch_basesize,
        threaded=threaded,
        kwargs...)
    
end

function run_simulation2d!(
    sim::Simulation{T},
    kybatches::Vector{Vector{T}};
    kxbatch_basesize=128,
    threaded=false,
    kwargs...) where {T<:Real}
    
    p                   = getparams(sim)
    observables_total   = deepcopy(sim.observables)
    nprogress           = nestedcount(kybatches) * p.nt * p.nkx
    current_progress    = 0.0

    @withprogress name="Simulation" for kybatch in kybatches
        observables_buffer = run_kybatch!(sim,kybatch;
            kxbatch_basesize=kxbatch_basesize,
            threaded=threaded,
            kwargs...)

        for (o,otot) in zip(observables_buffer,observables_total)
            addto!(o,otot)
        end

        resize_obs!(sim)
        
        @everywhere GC.gc()
        current_progress += p.nkx * p.nt * length(kybatch) / nprogress
        @debug "$(round(100*current_progress,digits=2))%"
        @logprogress current_progress
    end

    sim.observables .= observables_total

    return sim.observables
end

function run_kybatch!(
    sim::Simulation{T},
    kysamples::AbstractVector{T};
    threaded=false,
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}

    if threaded
        obs = Folds.map(
            ky -> run_simulation1d!(
                deepcopy(sim),
                ky;
                kxbatch_basesize=kxbatch_basesize,
                kwargs...),
            kysamples)
        integrateobs_threaded!(obs,sim.observables,kysamples)
    else
        obs = pmap(
            ky -> run_simulation1d!(
                sim,
                ky;
                kxbatch_basesize=kxbatch_basesize,
                kwargs...),
            kysamples)
        integrateobs!(obs,sim.observables,kysamples)
    end

    return sim.observables
end


function run_simulation_si!(
    sim::Simulation{T},
    kybatches::Vector{Vector{T}};
    savedata=true,
    saveplots=true,
    threaded=false,
    kxbatch_basesize=128,
    kwargs...) where {T<:Real}
    
    @info """
            ## $(getshortname(sim)) (id: $(sim.id))

            Starting on **$(gethostname())** at **$(now())**:
            
            * threads: $(Threads.nthreads())
            * processes: $(Distributed.nprocs())
            * maximum size of kx-batches: $kxbatch_basesize
            * maximum size of ky-batches: $(maximum(length.(kybatches)))
            * plotpath: $(sim.plotpath)
            * datapath: $(sim.datapath)

            $(markdown_paramsSI(sim))
            """

    checkbzbounds(sim)
    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    if sim.dimensions == 1
        run_simulation1d!(sim;kwargs...)
    else
        run_simulation2d!(sim,kybatches;
            threaded=threaded,
            kxbatch_basesize=kxbatch_basesize,
            kwargs...)
    end

    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end
    if saveplots
        plotdata(sim;kwargs...)
    end

    return sim.observables
end


function run_simulation_si!(ens::Ensemble;
    savedata=true,
    saveplots=true,
    ensembleparallel=false,
    threaded=false,
    maxparallel_ky=64,
    kxbatch_basesize=128,
    kwargs...)


    list_of_kybatches   = Vector{Vector{Vector{T}}}(undef,0)
    nprogress           = 0

    for p in getparams.(ens.simlist)
        kybatches = padvecto_overlap!(subdivide_vector(p.kysamples,maxparallel_ky))
        nprogress += nestedcount(kybatches) * p.nt * p.nkx
        push!(list_of_kybatches,kybatches) 
    end

    allobs              = []
    if ensembleparallel
        if threaded
            @warn "At the moment parallel ensembles are only supported via multi-processing\n"*
                    "Using pmap()"
        end
        allobs = pmap(
                (s,kys) -> run_simulation_si!(s,kys;
                    savedata=savedata,
                    saveplots=saveplots,
                    threaded=threaded,
                    kxbatch_basesize=kxbatch_basesize,
                    kwargs...),
                ens.simlist,list_of_kybatches)
        # Run plotting sequentially
        if saveplots
            Damysos.plotdata.(ens.simlist;kwargs...)
        end
        
    else
        current_progress = 0.0
        collection = zip(ens.simlist,list_of_kybatches,getparams.(ens.simlist))

        @withprogress name="Ensemble" for (s,kybatches,p) in collection
            obs = run_simulation_si!(s,kybatches;
                savedata=savedata,
                saveplots=saveplots,
                threaded=threaded,
                kxbatch_basesize=kxbatch_basesize,
                kwargs...)
            push!(allobs,obs)

            current_progress += nestedcount(kybatches) * p.nt * p.nkx / nprogress
            @logprogress current_progress
            @debug "$(round(100*current_progress,digits=2))%"
            @everywhere GC.gc
        end
    end
    
    return allobs
end