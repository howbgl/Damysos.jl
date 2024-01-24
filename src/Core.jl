
include("strategies/RowByRow.jl")
include("strategies/Tiling.jl")
include("strategies/LinearIndexing.jl")

"""
run_simulation!(sim::Simulation{T};
    savedata=true,
    saveplots=true,
    threaded=false,
    maxparallel_ky=64,
    kxbatch_basesize=512,
    kwargs...) where {T<:Real}

Run a simulation.

# Arguments
- `sim::Simulation{T}`: See [`Simulation`](@ref)
- `maxparallel_ky`: The maximum amount of different ky-lines computed in parallel. Good values are typically ~ 2nworkers. Large numbers mean high memory footprint.
- `kxbatch_basesize` : Number of kx modes per ky-line processed in one solve call. Large numbers mean high memory footprint.
- `kwargs...`: Additional keyword arguments are passed to the solve() function of DifferentialEquations.jl

# Returns
The observables obtained from the simulation.

# See also
[`run_simulation1d!`](@ref), [`run_simulation2d!`](@ref)

"""
function run_simulation!(
    sim::Simulation{T};
    savedata=true,
    saveplots=true,
    strategy="SolveAndIntegrate",
    threaded=false,
    kxbatch_basesize=128,
    maxparallel_ky=64,
    kwargs...) where {T<:Real}

    p = getparams(sim)
    if strategy=="SolveAndIntegrate"
        if sim.dimensions==1
            kybatches = Vector{Vector{T}}(undef,0)
        elseif  sim.dimensions==2
            kybatches = padvecto_overlap!(subdivide_vector(p.kysamples,maxparallel_ky))
        end
    
    
        return run_simulation_si!(sim,kybatches;
            savedata=savedata,
            saveplots=saveplots,
            threaded=threaded,
            kxbatch_basesize=kxbatch_basesize,
            kwargs...)
    elseif strategy=="SolveAndWrite"
        return solve_and_integrate_tiles!(sim;
            savedata=savedata,
            saveplots=saveplots,
            kwargs...)
    end
    
end


"""
    run_simulation!(ens::Ensemble{T};
        savedata=true,
        saveplots=true,
        ensembleparallel=false,
        threaded=false,
        maxparallel_ky=64,
        kxbatch_basesize=512,
        makecombined_plots=true,
        kwargs...) where {T<:Real}

Run simulations for an ensemble of `sim` objects.

# Arguments
- `ens::Ensemble{T}`: See [`Ensemble`](@ref)
- `maxparallel_ky`: The maximum amount of different ky-lines computed in parallel. Good values are typically ~ 2nworkers. Large numbers mean high memory footprint.
- `kxbatch_basesize` : Number of kx modes per ky-line processed in one solve call. Large numbers mean high memory footprint.
- `kwargs...`: Additional keyword arguments are passed to the solve() function of DifferentialEquations.jl

# Returns
An array of observables obtained from the simulations.

# See also
[`run_simulation2d!`](@ref), [`run_simulation!`](@ref)

"""
function run_simulation!(ens::Ensemble{T};
                savedata=true,
                saveplots=true,
                strategy="SolveAndIntegrate",
                ensembleparallel=false,
                threaded=false,
                maxparallel_ky=64,
                kxbatch_basesize=128,
                kwargs...) where {T<:Real}

    ensurepath(ens.datapath)
    ensurepath(ens.plotpath)

    @info """
        # Ensemble of $(length(ens.simlist)) Simulations
        
        * id : $(ens.id)
        * plotpath: $(ens.plotpath)
        * datapath: $(ens.datapath)"""


    if strategy=="SolveAndIntegrate"
        result = run_simulation_si!(
            ens;
            savedata=savedata,
            saveplots=saveplots,
            ensembleparallel=ensembleparallel,
            threaded=threaded,
            maxparallel_ky=maxparallel_ky,
            kxbatch_basesize=kxbatch_basesize,
            kwargs...)
    end
    

    if saveplots
        Damysos.plotdata(ens)
    end

    if savedata
        savemetadata(ens)
    end

    return result
end

