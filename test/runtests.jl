using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include("testsims.jl")

rm("testresults/", force = true, recursive = true)

global_logger(TerminalLogger(stderr, Logging.Info))

function checkvelocity(sim::Simulation, solver::DamysosSolver, fns, vref::Velocity;
	atol = 1e-10,
	rtol = 1e-2)
	res = run!(sim, fns, solver; saveplots = false)
	v   = filter(o -> o isa Velocity,res)[1]
	return isapprox(v, vref, atol = atol, rtol = rtol)
end

function makectest(sim::Simulation, m::ConvergenceTestMethod, s::Union{Nothing,DamysosSolver};
	atol::Real = 1e-10,
	rtol::Real = 1e-3)
	return isnothing(s) ? nothing : ConvergenceTest(sim, s, m, atol, rtol)
end

const sim1 = make_test_simulation1()
const sim1_dt = make_test_simulation1(0.08, 1.0, 1.0, 175, 2)
const sim1_kxmax = make_test_simulation1(0.01, 1.0, 1.0, 150, 2)

const linchunked = LinearChunked()
const fns1_linchunked = define_functions(sim1, linchunked)

const solver_autovern7 = LinearChunked(
	Damysos.default_kchunk_size(LinearChunked),
	Damysos.choose_threaded_or_distributed(),
	AutoVern7(KenCarp47(autodiff = false), lazy = true))
const fns_autovern7 = define_functions(sim1, solver_autovern7)
skipcuda = false

try
	LinearCUDA()
catch err
	if err == ErrorException("CUDA.jl is not functional, cannot use LinearCUDA solver.")
		global skipcuda = true
		@warn "Skipping CUDA tests, CUDA.jl is not functional."
	end
end
const lincuda = skipcuda ? nothing : LinearCUDA()
const fns1_lincuda = skipcuda ? nothing : define_functions(sim1, lincuda)


const referencedata = DataFrame(CSV.File("referencedata.csv"))
const vref = Velocity(
	referencedata.vx,
	referencedata.vxintra,
	referencedata.vxinter,
	referencedata.vy,
	referencedata.vyintra,
	referencedata.vyinter)


@testset "Damysos.jl" begin

	include("fieldtests.jl")
	include("matrixelements.jl")
	include("reference1d.jl")

	# @testset "Simulation 1 (2d)" begin
	# 	@testset "LinearCUDA" begin
	# 		@test checkvelocity(sim1, lincuda, fns1_lincuda, vref) skip = skipcuda
	# 	end
	# 	@testset "LinearChunked" begin
	# 		@test checkvelocity(sim1, linchunked, fns1_linchunked, vref)
	# 	end

	# 	@testset "AutoVern7" begin
	# 		@test checkvelocity(sim1, solver_autovern7, fns_autovern7, vref)
	# 	end
	# end

	

	# @testset "ConvergenceTest" begin
	# 	@testset "LinearChunked" begin
	# 		@testset "PowerLawTest (dt)" begin
	# 			ctest = makectest(sim1_dt, PowerLawTest(:dt, 0.5), linchunked)
	# 			@test successful_retcode(run!(ctest))
	# 		end
	# 		@testset "LinearTest (kxmax)" begin
	# 			ctest = makectest(sim1_kxmax, LinearTest(:kxmax, 10), linchunked)
	# 			@test successful_retcode(run!(ctest))
	# 		end
	# 		@testset "PowerLawTest (dkx)" begin
	# 			ctest = makectest(sim2_dkx, PowerLawTest(:dkx,0.5), linchunked)
	# 			@test successful_retcode(run!(ctest))
	# 		end
	# 	end
	# 	@testset "LinearCUDA" begin
	# 		@testset "PowerLawTest (dt)" begin
	# 			ctest = makectest(sim1_dt, PowerLawTest(:dt, 0.5), lincuda)
	# 			@test successful_retcode(run!(ctest)) skip = skipcuda
	# 		end
	# 		@testset "LinearTest (kxmax)" begin
	# 			ctest = makectest(sim1_kxmax, LinearTest(:kxmax, 10), lincuda)
	# 			@test successful_retcode(run!(ctest)) skip = skipcuda
	# 		end
	# 		@testset "PowerLawTest (dkx)" begin
	# 			ctest = makectest(sim2_dkx, PowerLawTest(:dkx,0.5), lincuda)
	# 			@test successful_retcode(run!(ctest)) skip = skipcuda
	# 		end
	# 	end
	# end
end

rm("testresults/", force = true, recursive = true)
