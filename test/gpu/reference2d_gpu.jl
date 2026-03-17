using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

const sim_2d = make_test_simulation_2d(; id = "sim2d_gpu")

skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

lincuda = skipcuda ? nothing : LinearCUDA(10_000, GPUVern7(), 1)
const fns_2d_lincuda = skipcuda ? nothing : define_functions(sim_2d, lincuda)

@testset "Reference (2d)" begin
    @testset "LinearCUDA" begin
        @test test_2d(vref2d, sim_2d, fns_2d_lincuda, lincuda) skip = skipcuda
    end
end
