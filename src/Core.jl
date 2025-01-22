

export define_functions
export run!

include("solvers/LinearChunked.jl")
include("solvers/LinearCUDA.jl")
include("solvers/SingleMode.jl")

DEFAULT_REDUCTION(u, data, I) = (append!(u,sum(data)),false)

default_kchunk_size(::Type{T}) where {T<:DamysosSolver} = 256


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
- `savepath::String`: path to directory to save data & plots
- `showinfo::Bool`: log/display simulation info before running
- `nan_limit::Int`: maximum tolerated number of nans in observables

# Returns
The observables obtained from the simulation.

# See also
[`Simulation`](@ref), [`define_functions`](@ref), [`LinearChunked`](@ref)

"""
function run!(sim::Simulation,functions,solver::DamysosSolver=LinearChunked(); kwargs...)

    prerun!(sim,solver;kwargs...)
    _run!(sim,functions,solver)
    postrun!(sim;kwargs...)

    return sim.observables
end

"""
    define_functions(sim,solver)

Hardcode the functions needed to run the Simulation. 

# Arguments
- `sim::Simulation`: contains physical & numerical information (see [`Simulation`](@ref))
- `solver`: strategy for integrating in k-space. Defaults to [`LinearChunked`](@ref))

# Returns
Vector of functions used by [`run!`](@ref).

# See also
[`Simulation`](@ref), [`run!`](@ref), [`LinearChunked`](@ref)

"""
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
    
    p   = sim.numericalparams
    Δk  = if sim.dimensions == 2
            p.dkx * p.dky
        elseif sim.dimensions == 1
            p.dkx 
        elseif sim.dimensions == 0
            1.0
        end
    normalize!.(sim.observables,(2π)^sim.dimensions / Δk)

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
