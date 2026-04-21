export PreparedSimulation

"""
    PreparedSimulation(sim[, solver])

Bundles a [`Simulation`](@ref), a `DamysosSolver`, and the corresponding
`SimulationFunctions` for execution via [`run!`](@ref).

Note that the call to `run!(psim)` must be more recent in world age than the definition of 
the `PreparedSimulation`.

# Avialable solvers
- [`LinearCUDA()`](@ref)
- [`LinearChunked()`](@ref)
- [`SingleMode()`](@ref)

# See also
[`run!`](@ref), [`Simulation`](@ref)
"""
struct PreparedSimulation{T <: Real, U}
    sim::Simulation{T}
    solver::DamysosSolver
    functions::SimulationFunctions{U}
end

function PreparedSimulation(
    sim::Simulation,
    solver::DamysosSolver = LinearChunked())

    functions = define_functions(sim, solver)
    return PreparedSimulation(sim, solver, functions)
end
