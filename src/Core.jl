

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

# Returns
The observables obtained from the simulation.

# See also
[`Simulation`](@ref), [`define_functions`](@ref), [`LinearChunked`](@ref)

"""
function run!(sim::Simulation,functions,solver::DamysosSolver=LinearChunked();
    savedata=true,
    saveplots=true,
    showinfo=true)

    prerun!(sim,solver;savedata=savedata,saveplots=saveplots,showinfo=showinfo)
    _run!(sim,functions,solver)
    postrun!(sim;savedata=savedata,saveplots=saveplots)

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
    showinfo=true)

    !solver_compatible(sim,solver) && throw(incompatible_solver_exception(sim,solver))
    showinfo && printinfo(sim,solver)
    
    checkbzbounds(sim)
    savedata && ensurepath(sim.datapath)
    saveplots && ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)

    return nothing
end

function postrun!(sim::Simulation;savedata=true,saveplots=true)
    
    p   = sim.numericalparams
    Δk  = if sim.dimensions == 2
            p.dkx * p.dky
        elseif sim.dimensions == 1
            p.dkx 
        elseif sim.dimensions == 0
            1.0
        end
    normalize!.(sim.observables,(2π)^sim.dimensions / Δk)

    savedata && Damysos.savedata(sim)
    saveplots && plotdata(sim)
    
    return nothing
end

function runtimeout!(timeout,sim::Simulation,fns,solver::DamysosSolver;savedata=true,saveplots=true)

    _run = let sim=sim,fns=fns,solver=solver,sd=savedata,sp=saveplots
        c -> begin
            try
                put!(c,run!(sim,fns,solver;savedata=sd,saveplots=sp))
                return nothing
            catch e
                if e isa InterruptException
                    @warn "run interrupted!"
                end
            end
        end 
    end

    runtask = @async begin
        try
            run!(sim,fns,solver;savedata=savedata,saveplots=saveplots)
        catch e
            if e isa InterruptException
                @warn "run interrupted!"
            end
        end
    end

    sig = timedwait(()->istaskdone(runtask),timeout)
    sig == :timed_out && schedule(runtask,InterruptException(),error=true)
    return sim.observables
end

function printinfo(sim::Simulation,solver::DamysosSolver)
    @info """
        ## $(getshortname(sim)) (id: $(sim.id))

        Starting on **$(gethostname())** at **$(now())**:
        
        * threads: $(Threads.nthreads())
        * processes: $(Distributed.nprocs())
        * Solver: $(repr(solver))
        * plotpath: $(sim.plotpath)
        * datapath: $(sim.datapath)

        $(markdown_paramsSI(sim))
        """
end
