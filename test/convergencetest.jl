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
	loaded_test = ConvergenceTest(path,LinearCUDA(),resume=true)
	return all(test.completedsims .≈ loaded_test.completedsims)
end

const sim = make_test_simulation_tiny()

linchunked = LinearChunked(256,EnsembleThreads(),Vern7(),1e-8,0.01)

skipcuda = false

try
	LinearCUDA()
catch err
	if err == ErrorException("CUDA.jl is not functional, cannot use LinearCUDA solver.")
		global skipcuda = true
		@warn "Skipping CUDA tests, CUDA.jl is not functional."
	end
end
lincuda = skipcuda ? nothing : LinearCUDA(10_000,GPUVern7(),1)

const dt_tests = [ConvergenceTest(
    sim,
    solver;
    method = PowerLawTest(:dt,0.5),
    atolgoal = 1e-6,
    rtolgoal = 1e-2,
    maxtime = u"24*60minute",
    maxiterations = 20) for solver in filter(x->!isnothing(x),(linchunked,lincuda))]


@testset "ConvergenceTest" begin
    @testset "LinearChunked" begin
        @testset "dt" begin
            @test successful_retcode(run!(dt_tests[1];filepath="testresults/dt_linchunked.hdf5"))
        end
    end
    @testset "LinearCUDA" begin
        @testset "dt" begin
            @test successful_retcode(run!(dt_tests[2];filepath="testresults/dt_cuda.hdf5")) skip = skipcuda
        end
    end
end

@testset "Plotting" begin
	@test test_plotting_simvector(dt_tests[1].completedsims)
end

@testset "HDF5 loading" begin
	@test test_continue_ctest(dt_tests[1],"testresults/dt_linchunked.hdf5")
end
