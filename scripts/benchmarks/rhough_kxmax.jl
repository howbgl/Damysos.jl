using Damysos, TerminalLoggers, Dates, LoggingExtras

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function make_simulation(M = 0.5, ζ = 4.0, datapath = "~/data", plotpath = datapath)

	ħ = Unitful.ħ
	vf = u"497070.0m/s"
	freq = u"5.0THz"
	# emax      = u"0.5MV/cm"
	e = uconvert(u"C", 1u"eV" / 1u"V")
	ω = 2π * freq
	m = M * ħ * ω / 2

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", ζ * ħ * ω^2 / (2vf * e))
	us = scaledriving_frequency(freq, vf)

	t2 = Inf * u"s"
	t1 = Inf * u"s"

	h  = GappedDirac(energyscaled(m, us))
	l  = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df = GaussianAPulse(us, σ, freq, emax)

	axmax = df.eE / df.ω

	dt    = 2e-5
	kxmax = 3axmax
	dkx   = 2kxmax / 1_000
	kymax = 1.0
	dky   = 1.0

	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_rough_kxmax_1e-2"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function make_kxmax_test(sim::Simulation,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	kxmax  = sim.numericalparams.kxmax
	method = LinearTest(:kxmax,0.2kxmax)
	atolgoal = 1e-12
	rtolgoal = 1e-2
	maxtime = u"120minute"
	maxiter = 16 
	return ConvergenceTest(sim,solver,method,atolgoal,rtolgoal,maxtime,maxiter;
		altpath = altpath)
end

const Ms = exp10.(LinRange(-1,1,3))
const ζs = exp10.(LinRange(-1,1,3))
const id = parse(Int,ENV["SLURM_JOB_ID"])

path(ζ) = joinpath(ENV["WORK"], "rhough_kxmax/1e-2/zeta=$(ζ)")
altpath(ζ) = joinpath(ENV["HPCVAULT"], "rhough_kxmax/1e-2/zeta=$(ζ)")
const sims = [make_simulation(m,ζ,path(ζ)) for m in Ms for ζ in ζs]
const tests = [make_kxmax_test(s) for s in sims]

const logger = make_teelogger(
	joinpath(ENV["WORK"],"rhough_kxmax/1e-2"), 
	"slurmid="*ENV["SLURM_JOB_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

run!.(tests)

@info "$(now()): calculation finished."
