using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function makesim(M = 0.5, ζ = 4.0, dt = 0.01; datapath = "~/data", plotpath = datapath)

	freq      	   = u"5.0THz"
	ω 			   = 2π*freq
	effective_mass = 0.067 * m_e # rhough lit value for GaAs
	gap 		   = uconvert(u"eV",M * ħ*ω)

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", sqrt(ζ * ħ * ω^3 * effective_mass / q_e^2))
	
	# choose lengthscale such that the internal ζ = our non-dim ζ
	lc = uconvert(u"nm",sqrt(2π) * ħ * ω / (q_e * emax))
	us = UnitScaling(1/freq,lc)

	t2 = uconvert(u"fs", 5 / freq)
	t1 = Inf * u"s"

	h    = QuadraticToy(us,gap,effective_mass)
	l    = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)

	axmax = df.eE / df.ω

	kxmax = 3axmax
	dkx   = 2kxmax / 1_000
	kymax = 1.0
	dky   = 1.0

	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_dkx"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function maketest(sim::Simulation,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dkx,0.5)
	atolgoal = 1e-12
	rtolgoal = 1e-4
	maxtime = u"10*60minute"
	maxiter = 40 
	return ConvergenceTest(sim,solver,method,atolgoal,rtolgoal,maxtime,maxiter;
		altpath = altpath)
end

const id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const start = CSV.File(joinpath(
	ENV["WORK"],
	"quadratic/zeta-M-sweep2/dt1/results.csv"))

const zeta 	= start.zeta[id]
const m 	= start.M[id]
const dt 	= start.dt[id]

subdir 			= "quadratic/zeta-M-sweep2/dkx1"
path(ζ,M) 		= joinpath(ENV["WORK"], subdir, "zeta=$(ζ)_M=$(M)")
altpath(ζ,M) 	= joinpath(ENV["HPCVAULT"], subdir, "zeta=$(ζ)_M=$(M)")

const sim 	= makesim(m, zeta, dt; datapath = path(zeta,m))
const test 	= maketest(sim, altpath(zeta,m))

const logger = make_teelogger(
	joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)