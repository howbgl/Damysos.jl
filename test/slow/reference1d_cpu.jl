using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

const sim_1d = make_test_simulation_1d(; id = "sim1d_cpu")
linchunked = LinearChunked()
const fns_1d_linchunked = define_functions(sim_1d, linchunked)

@testset "Reference (1d)" begin
    @testset "LinearChunked" begin
        @test test_1d(vref1d, sim_1d, fns_1d_linchunked, linchunked)
    end
end
