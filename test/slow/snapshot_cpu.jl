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

function test_snapshots_saving_loading(sim::Simulation)
    path = joinpath(testresults_dir(), "snapshot_test")
    Damysos.savedata(sim, path)
    return isapprox(sim, Simulation(joinpath(path, "data.hdf5")))
end

const sim_snap = make_test_simulation_snap()

linchunked = LinearChunked(256)
const fns_sim_snap_linchunked = define_functions(sim_snap, linchunked)

@testset "DensityMatrixSnapshots" begin
    @testset "LinearChunked" begin
        @test test_snapshots(sim_snap, fns_sim_snap_linchunked, linchunked)
    end
    @testset "Saving & loading" begin
        @test test_snapshots_saving_loading(sim_snap)
    end
end
