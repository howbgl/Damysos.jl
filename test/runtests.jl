using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

global_logger(TerminalLogger(stderr,Logging.Warn))

function make_test_simulation1(
	dt::Real = 0.01,
	dkx::Real = 1.0,
	dky::Real = 1.0,
	kxmax::Real = 175,
	kymax::Real = 100)

	vf     = u"4.3e5m/s"
	freq   = u"5THz"
	m      = u"20.0meV"
	emax   = u"0.1MV/cm"
	tcycle = uconvert(u"fs", 1 / freq) # 100 fs
	t2     = tcycle / 4             # 25 fs
	t1     = Inf * u"1s"
	σ      = u"800.0fs"

	# converged at
	# dt = 0.01
	# dkx = 1.0
	# dky = 1.0
	# kxmax = 175
	# kymax = 100

	us   = scaledriving_frequency(freq, vf)
	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)
	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id    = "sim1"
	dpath = "testresults/sim1"
	ppath = "testresults/sim1"

	return Simulation(l, df, pars, obs, us, id, dpath, ppath)
end

function checkvelocity(sim::Simulation, solver::DamysosSolver, fns, vref::Velocity;
	atol = 1e-10,
	rtol = 1e-2)
	res = run!(sim, fns, solver; saveplots = false)
	v   = sim1.observables[1]
	return isapprox(v, vref, atol = atol, rtol = rtol)
end

const sim1 = make_test_simulation1()

const linchunked = LinearChunked()
const fns_linchunked = define_functions(sim1, linchunked)

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
const fns_lincuda = skipcuda ? nothing : define_functions(sim1, lincuda)

const referencedata = DataFrame(CSV.File("referencedata.csv"))
const vref = Velocity(
	referencedata.vx,
	referencedata.vxintra,
	referencedata.vxinter,
	referencedata.vy,
	referencedata.vyintra,
	referencedata.vyinter)

@testset "Damysos.jl" begin
	@testset "Simulation 1" begin
		@testset "LinearCUDA" begin
			@test checkvelocity(sim1, lincuda, fns_lincuda, vref) skip = skipcuda
		end
		@testset "LinearChunked" begin
			@test checkvelocity(sim1, linchunked, fns_linchunked, vref)
		end
	end

	@testset "ConvergenceTest" begin
		@testset "PowerLawTest (dt)" begin
			sim_dt = make_test_simulation1(0.08, 1.0, 1.0, 175, 2)
			convergence_test =
				ConvergenceTest(sim_dt, linchunked, PowerLawTest(:dt, 0.5), 1e-10, 1e-3)
			res = run!(convergence_test)
			@test res.success
		end
		@testset "LinearTest (kxmax)" begin
			sim_kxmax = make_test_simulation1(0.01, 1.0, 1.0, 150, 2)
			convergence_test =
				ConvergenceTest(sim_kxmax, linchunked, LinearTest(:kxmax, 10), 1e-10, 1e-3)
			res = run!(convergence_test)
			@test res.success			
		end
	end
end
