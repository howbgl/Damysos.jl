"""
    PreparedSimulation(sim[, solver])
    PreparedSimulation(sim, solver, functions)

Bundle a [`Simulation`](@ref), a [`DamysosSolver`](@ref), and the corresponding
[`SimulationFunctions`](@ref) for execution via [`run!`](@ref).

The preferred construction path is `PreparedSimulation(sim, solver)`, which
computes the solver-specific function bundle with [`define_functions`](@ref).
The 3-argument form remains available for advanced callers that already have a
prepared `SimulationFunctions` instance.
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
