using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

const sim_2d = make_test_simulation_2d(; id = "sim2d_multigpu")

multigpu = false
skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

if !skipcuda
    s = LinearCUDA()
    global multigpu = s.ngpus > 1
end

s_multigpu = multigpu ? LinearCUDA() : nothing
const fns_2d_multigpu = multigpu ? define_functions(sim_2d, s_multigpu) : nothing

@testset "Reference (2d)" begin
    @testset "LinearCUDA: multi-GPU" begin
        @test test_2d(vref2d, sim_2d, fns_2d_multigpu, s_multigpu) skip = !multigpu
    end
end
