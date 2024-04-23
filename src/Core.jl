

export define_functions
export run!

include("solvers/LinearChunked.jl")
include("solvers/LinearCUDA.jl")

DEFAULT_REDUCTION(u, data, I) = (append!(u,sum(data)),false)


"""
    run!(sim, functions[, solver]; kwargs...)

Run a simulation.

# Arguments
- `sim::Simulation`: contains physical & numerical information (see [`Simulation`](@ref))
- `functions`: needed by the solver/integrator (see [`define_functions`](@ref)).
- `solver`: strategy for integrating in k-space. Defaults to [`LinearChunked`](@ref))

# Keyword Arguments
- `savedata::Bool`: save observables and simulation to disk after completion
- `plotdata::Bool`: create default plots and save them to disk after completion

# Returns
The observables obtained from the simulation.

# See also
[`Simulation`](@ref), [`define_functions`](@ref), [`LinearChunked`](@ref)

"""
function run!(
    sim::Simulation,
    functions,
    solver::DamysosSolver=LinearChunked();
    savedata=true,
    saveplots=true)
    
    run!(
        sim,
        functions,
        solver,
        savedata=savedata,
        saveplots=saveplots)
end

"""
    define_functions(sim[, solver])

Hardcode the functions needed to run the Simulation. 

# Arguments
- `sim::Simulation`: contains physical & numerical information (see [`Simulation`](@ref))
- `solver`: strategy for integrating in k-space. Defaults to [`LinearChunked`](@ref))

# Returns
Vector of functions used by [`run!`](@ref).

# See also
[`Simulation`](@ref), [`run!`](@ref), [`LinearChunked`](@ref)

"""
define_functions(sim::Simulation) = define_functions(sim,LinearChunked())

function prerun!(sim::Simulation;savedata=true,saveplots=true)

    @info """
        ## $(getshortname(sim)) (id: $(sim.id))

        Starting on **$(gethostname())** at **$(now())**:
        
        * threads: $(Threads.nthreads())
        * processes: $(Distributed.nprocs())
        * plotpath: $(sim.plotpath)
        * datapath: $(sim.datapath)

        $(markdown_paramsSI(sim))
        """

    checkbzbounds(sim)
    savedata && ensurepath(sim.datapath)
    saveplots && ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    return nothing
end

function postrun!(sim::Simulation;savedata=true,saveplots=true)
    
    normalize!.(sim.observables,(2π)^sim.dimensions)

    savedata && Damysos.savedata(sim)
    saveplots && plotdata(sim)
    
    return nothing
end
