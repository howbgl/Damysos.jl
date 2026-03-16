using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

const sim_1d = make_test_simulation_1d(; id = "sim1d_gpu")
skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."
lincuda = skipcuda ? nothing : LinearCUDA(10_000, GPUVern7(), 1)
const fns_1d_lincuda = skipcuda ? nothing : define_functions(sim_1d, lincuda)

@testset "Reference (1d)" begin
    @testset "LinearCUDA" begin
        @test test_1d(vref1d, sim_1d, fns_1d_lincuda, lincuda) skip = skipcuda
    end
end
