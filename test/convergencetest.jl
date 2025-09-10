using CUDA
using CSV
using Damysos
using DataFrames
using HDF5
using LoggingExtras
using TerminalLoggers
using Test

import Damysos.load_obj_hdf5

function make_test_simulation_tiny(
	dt::Real = 0.01,
	dkx::Real = 2.0,
	dky::Real = 1.0,
	kxmax::Real = 175,
	kymax::Real = 10)

	vf     = u"4.3e5m/s"
	freq   = u"5THz"
	m      = u"20.0meV"
	emax   = u"0.1MV/cm"
	tcycle = uconvert(u"fs", 1 / freq) # 100 fs
	t2     = tcycle / 4             # 25 fs
	t1     = Inf * u"1s"
	σ      = u"800.0fs"

	us   = scaledriving_frequency(freq, vf)
	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)
	tgrid = SymmetricTimeGrid(dt, -5df.σ)
	kgrid = CartesianKGrid2d(dkx, kxmax, dky, kymax)
	grid = NGrid(kgrid,tgrid)
	obs  = [Velocity(grid), Occupation(grid)]

	id    = "sim1"

	return Simulation(l, df, grid, obs, us, id)
end

function test_plotting_simvector(sims::Vector{Simulation})
	path = "testresults/plots"
	plotdata(sims, path)
	return any([occursin(".png",f) for f in readdir(path,join=true)])
end

function test_continue_ctest(test::ConvergenceTest,path::String)
	loaded_test = ConvergenceTest(path,LinearCUDA())
	return all(test.completedsims .≈ loaded_test.completedsims)
end

function test_extend_kymaxtest(test, extend_test)
	
	run!(test; savedata=false)
	run!(extend_test; savedata=false)
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
										filepath="testresults/dt_linchunked.hdf5"))
        end
    end
    @testset "LinearCUDA" begin
        @testset "dt" begin
            @test successful_retcode(run!(dt_tests[2];
									filepath="testresults/dt_cuda.hdf5")) skip = skipcuda
        end
    end
	@testset "Plotting" begin
		@test test_plotting_simvector(dt_tests[1].completedsims)
	end
	@testset "HDF5 loading" begin
		@test test_continue_ctest(dt_tests[1],"testresults/dt_linchunked.hdf5")
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

