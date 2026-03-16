using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

function test_composite_1d(sim::Simulation,fns,solver::DamysosSolver;
    atol = 1e-10,
    rtol = 1e-2)

    run!(sim, fns, solver; saveplots = false, savedata = true, showinfo = false,
        savepath = joinpath(testresults_dir(), sim.id))
    return true
end

function test_load_composite_1d(sim::Simulation, path::String; atol = 1e-10, rtol = 1e-8)
    loaded_sim = Simulation(path)
    return isapprox(sim, loaded_sim, atol = atol, rtol = rtol)
end

const sim_composite_1d = make_test_simulation_composite_1d(; id = "sim1d_composite_cpu")

linchunked = LinearChunked()
const fns_1d_linchunked = define_functions(sim_composite_1d, linchunked)

@testset "CompositeField" begin
    @testset "LinearChunked" begin
        @test test_composite_1d(sim_composite_1d, fns_1d_linchunked, linchunked)
    end
    @testset "Loading from .hdf5" begin
        @test test_load_composite_1d(
            sim_composite_1d,
            joinpath(testresults_dir(), sim_composite_1d.id, "data.hdf5"))
    end
end
