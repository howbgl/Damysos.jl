using CUDA
using CSV
using Damysos
using DataFrames
using HDF5
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))
include(joinpath(@__DIR__, "..", "test_simulations.jl"))

import Damysos.load_obj_hdf5

const sim_snap = make_test_simulation_snap()

skipcuda = !(CUDA.functional())

skipcuda && @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."
lincuda = skipcuda ? nothing : LinearCUDA(10_000)
const fns_sim_snap_lincuda = skipcuda ? nothing : define_functions(sim_snap, lincuda)

@testset "DensityMatrixSnapshots (GPU)" begin
    @testset "LinearCUDA" begin
        @test test_snapshots(sim_snap, fns_sim_snap_lincuda, lincuda) skip = skipcuda
    end
end
