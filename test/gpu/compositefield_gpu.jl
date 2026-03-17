using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

const sim_composite_1d_gpu = make_test_simulation_composite_1d(; id = "sim1d_composite_gpu")

skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

lincuda = skipcuda ? nothing : LinearCUDA(10_000, GPUVern7(), 1)
const fns_1d_lincuda = skipcuda ? nothing : define_functions(sim_composite_1d_gpu, lincuda)

@testset "CompositeField (GPU)" begin
    @testset "LinearCUDA" begin
        @test !isempty(run!(sim_composite_1d_gpu, fns_1d_lincuda, lincuda;
            showinfo = false, 
            saveplots = false, 
            savedata = false)) skip = skipcuda
    end
end
