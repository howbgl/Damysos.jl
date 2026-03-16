using CUDA
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

const sim_composite_1d = make_test_simulation_composite_1d(; id = "sim1d_composite_gpu")

skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

lincuda = skipcuda ? nothing : LinearCUDA(10_000, GPUVern7(), 1)
const fns_1d_lincuda = skipcuda ? nothing : define_functions(sim_composite_1d, lincuda)

@testset "CompositeField (GPU)" begin
    @testset "LinearCUDA" begin
        @test test_composite_1d(sim_composite_1d, fns_1d_lincuda, lincuda) skip = skipcuda
    end
end
