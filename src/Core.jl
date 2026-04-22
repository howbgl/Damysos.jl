

export define_functions
export run!

include("solvers/LinearChunked.jl")
include("solvers/LinearCUDA.jl")
include("solvers/SingleMode.jl")

DEFAULT_REDUCTION(u, data, I) = (append!(u,sum(data)),false)

default_kchunk_size(::Type{T}) where {T<:DamysosSolver} = 256


"""
    run!(sim; kwargs...)

Run a simulation given as Simulation or PreparedSimulation.

# Keyword Arguments
- `solver::DamysosSolver`: only possible if `sim` is a `Simulation`
- `savedata::Bool`: save observables and simulation to disk after completion
- `saveplots::Bool`: create default plots and save them to disk after completion
- `savepath::String`: path to directory to save data & plots
- `showinfo::Bool`: log/display simulation info before running
- `nan_limit::Int`: maximum tolerated number of nans in observables

# Returns
The observables obtained from the simulation.

# See also
[`PreparedSimulation`](@ref), [`Simulation`](@ref)

"""
function run!(psim::PreparedSimulation; kwargs...)
    return _run_prepared!(psim.sim, psim.functions, psim.solver; kwargs...)
end

function run!(sim::Simulation; solver::DamysosSolver=LinearChunked(), kwargs...)
    return invokelatest(run!, PreparedSimulation(sim, solver); kwargs...)
end

function run!(sim::Simulation, functions::SimulationFunctions,
	solver::DamysosSolver = LinearChunked(); kwargs...)
    return run!(PreparedSimulation(sim, solver, functions); kwargs...)
end

function _run_prepared!(
    sim::Simulation,
    functions::SimulationFunctions,
    solver::DamysosSolver;
    kwargs...)

    prerun!(sim,solver;kwargs...)
    _run!(sim,functions,solver)
    postrun!(sim;kwargs...)

    return sim.observables
end


function define_functions end

function prerun!(sim::Simulation,solver::DamysosSolver;
    savedata=true,
    saveplots=true,
    savepath=joinpath(pwd(),getname(sim)),
    showinfo=true,
    kwargs...)

    !solver_compatible(sim,solver) && throw(incompatible_solver_exception(sim,solver))
    showinfo && printinfo(sim,solver)
    
    checkbzbounds(sim)
    if savedata || saveplots
        ensuredirpath(savepath)
    end

    resize_obs!(sim)
    zero.(sim.observables)

    return nothing
end

function postrun!(sim::Simulation;
    savedata=true,
    saveplots=true,
    savepath=joinpath(pwd(),getname(sim)),
    nan_limit=DEFAULT_NAN_LIMIT,
    kwargs...)
    
    applyweights_afterintegration!(sim.observables, sim.grid.kgrid)
    normalize!.(sim.observables,(2π)^sim.dimensions)

    nancount = count_nans(sim.observables)
    nancount > nan_limit && @warn "Too many Nans ($nancount)!"

    savedata && Damysos.savedata(sim,savepath)
    saveplots && plotdata(sim,savepath)
    
    return nothing
end

function printinfo(sim::Simulation,solver::DamysosSolver)
    @info """
        ## $(getshortname(sim)) (id: $(sim.id))

        Starting on **$(gethostname())** at **$(now())**:
        
        * threads: $(Threads.nthreads())
        * processes: $(Distributed.nprocs())
        * Solver: $(repr(solver))

        $(markdown_paramsSI(sim))
        """
end
