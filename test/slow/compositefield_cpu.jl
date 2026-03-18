using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

function test_load_composite_1d(sim::Simulation, path::String; atol = 1e-10, rtol = 1e-8)
    loaded_sim = Simulation(path)
    return isapprox(sim, loaded_sim, atol = atol, rtol = rtol)
end

const sim_composite_1d_cpu = make_test_simulation_composite_1d(; id = "sim1d_composite_cpu")

linchunked = LinearChunked()
const fns_composite_1d_linchunked = define_functions(sim_composite_1d_cpu, linchunked)

@testset "CompositeField" begin
    savepath_cpu_composite_1d = joinpath(
        testresults_dir(), sim_composite_1d_cpu.id, "cpu_composite_1d")
    @testset "LinearChunked" begin
        @test !isempty(run!(sim_composite_1d_cpu, fns_composite_1d_linchunked, linchunked;
            showinfo = false, 
            saveplots = false, 
            savedata = true, 
            savepath = savepath_cpu_composite_1d))
    end
    @testset "Loading from .hdf5" begin
        @test test_load_composite_1d(
            sim_composite_1d_cpu, joinpath(savepath_cpu_composite_1d, "data.hdf5"))
    end
end
