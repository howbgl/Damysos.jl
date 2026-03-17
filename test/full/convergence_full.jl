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


function test_plotting_simvector(sims::Vector{Simulation})
	path = joinpath(testresults_dir(), "plots")
	mkpath(path)
	plotdata(sims, path)
	return any([occursin(".png",f) for f in readdir(path,join=true)])
end

function test_continue_ctest(test::ConvergenceTest,path::String)
	loaded_test = ConvergenceTest(path,LinearCUDA())
	return all(test.completedsims .≈ loaded_test.completedsims)
end

function test_extend_kymaxtest(test, extend_test)
	
	run!(test; savedata=false, showinfo=false)
	run!(extend_test; savedata=false, showinfo=false)
	return all(test.completedsims .≈ extend_test.completedsims)
end

const sim = make_test_simulation_tiny()

linchunked = LinearChunked(256,EnsembleThreads(),Vern7(),1e-8,0.01)

skipcuda = !(CUDA.functional())

skipcuda &&  @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."
lincuda = skipcuda ? nothing : LinearCUDA(10_000,GPUVern7(),1)

const dt_tests = [ConvergenceTest(
    sim,
    solver;
    method = PowerLawTest(:dt,0.5),
    atolgoal = 1e-6,
    rtolgoal = 1e-2,
    maxtime = Inf,
    maxiterations = 20) for solver in filter(x->!isnothing(x),(linchunked,lincuda))]

const kymax_tests = [ConvergenceTest(
	sim,
	solver;
	method = PowerLawTest(:kymax,1.5),
	atolgoal = 1e-6,
	rtolgoal = 1e-2,
	maxtime = u"60minute",
	maxiterations = 4) for solver in filter(x->!isnothing(x),(linchunked,lincuda))]

const kymax_tests_extend = [ConvergenceTest(
	sim,
	solver;
	method = ExtendKymaxTest(PowerLawTest(:kymax,1.5)),
	atolgoal = 1e-6,
	rtolgoal = 1e-2,
	maxtime = u"60minute",
	maxiterations = 4) for solver in filter(x->!isnothing(x),(linchunked,lincuda))]
	
@testset "ConvergenceTest" begin
    @testset "LinearChunked" begin
        @testset "dt" begin
            @test successful_retcode(run!(dt_tests[1];
                                        showinfo=false,
                                        filepath=joinpath(testresults_dir(), "dt_linchunked.hdf5")))
        end
    end
    @testset "LinearCUDA" begin
        @testset "dt" begin
            @test successful_retcode(run!(dt_tests[2];
                                    showinfo=false,
                                    filepath=joinpath(testresults_dir(), "dt_cuda.hdf5"))) skip = skipcuda
        end
    end
	@testset "Plotting" begin
		@test test_plotting_simvector(dt_tests[1].completedsims)
	end
	@testset "HDF5 loading" begin
		@test test_continue_ctest(dt_tests[1], joinpath(testresults_dir(), "dt_linchunked.hdf5"))
	end
	@testset "ExtendKymaxTest" begin
		@testset "LinearChunked" begin
			# Something is wrong with LinearChunked, the error is small for this example
			# (≈ 1e-7) but for LinearCUDA its ≈ 1e-15 as expected. Plotting the error also
			# reveals it to show some structure from the pulse for some reason, so it seams
			# systematic and not floating point rounding error.
			@test test_extend_kymaxtest(kymax_tests[1], kymax_tests_extend[1]) broken=true
		end
		@testset "LinearCUDA" begin
			@test test_extend_kymaxtest(kymax_tests[2], kymax_tests_extend[2])
		end
	end
end
