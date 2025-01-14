using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV

import Damysos.getname

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function makesim(M = 0.5, ζ = 4.0, datapath = "~/data", plotpath = datapath)

	freq      	   = u"5.0THz"
	ω 			   = 2π*freq
	effective_mass = 0.067 * m_e # rhough lit value for GaAs
	gap 		   = uconvert(u"eV",M * ħ*ω)

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", sqrt(ζ * ħ * ω^3 * effective_mass / q_e^2))
	
	# choose lengthscale such that the internal ζ = our non-dim ζ
	lc = uconvert(u"nm",sqrt(2π) * ħ * ω / (q_e * emax))
	us = UnitScaling(1/freq,lc)

	t2 = Inf * u"s"
	t1 = Inf * u"s"

	h    = QuadraticToy(us,gap,effective_mass)
	l    = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)

	axmax = df.eE / df.ω

	dt    = 0.1
	kxmax = 3axmax
	dkx   = 2kxmax / 500
	kymax = kxmax / 10
	dky   = 2kymax / 32

	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_dt"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function maketest(sim::Simulation,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dt,0.5)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-12,
		rtolgoal = 1e-4,
		maxtime = u"60minute",
		maxiterations = 42,
		path 	= joinpath(sim.datapath, "convergencetest_$(getname(method))_1e-4.hdf5"),
		altpath = altpath)
end

const Ms = [0.1,1.0,10.0]
const ζs = [0.1,1.0,10.0]
const id = parse(Int,ENV["SLURM_ARRAY_TASK_ID"])

subdir 			= "quadratic/zeta-M-sweep1/dt_1e-4"
path(ζ,M) 		= joinpath(ENV["WORK"], subdir, "zeta=$(ζ)_M=$(M)")
altpath(ζ,M) 	= joinpath(ENV["HPCVAULT"], subdir, "zeta=$(ζ)_M=$(M)")
const tests		= [maketest(makesim(M,ζ,path(ζ,M)),altpath(ζ,M)) for ζ in ζs for M in Ms]

const logger = make_teelogger(
	joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = run!(tests[id])

@info "$(now()): calculation finished."

exit(res.retcode)