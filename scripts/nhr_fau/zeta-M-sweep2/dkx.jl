using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function makesim(M = 0.5, ζ = 4.0, dt = 0.1; 
	datapath = "~/data", 
	plotpath = datapath)

	ħ  = Unitful.ħ
	vf = u"497070.0m/s"
	freq      = u"5.0THz"
	e = uconvert(u"C", 1u"eV" / 1u"V")
	ω = 2π*freq
	m = M * ħ * ω / 2

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", ζ * ħ * ω^2 / (2vf * e))
	us = scaledriving_frequency(freq, vf)

	t2 = uconvert(u"fs", 5 / freq)
	t1 = Inf * u"s"

	h    = GappedDirac(energyscaled(m, us))
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
	rtolgoal = 1e-6
	maxtime = u"2*60minute"
	maxiter = 32 # min dkx = 0.01/2^31 ≈ 5e-12
	return ConvergenceTest(sim,solver,method,atolgoal,rtolgoal,maxtime,maxiter;
		altpath = altpath)
end

const id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const start = CSV.File(joinpath(
	ENV["WORK"],
	"zeta-M-sweep2/dt1/dt1-conv.csv"))

const zeta 	= start.zeta[id]
const m 	= start.M[id]
const dt 	= start.dt[id]

subdir 			= "zeta-M-sweep2/dkx1"
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