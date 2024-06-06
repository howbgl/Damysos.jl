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

	ħ  = Unitful.ħ
	vf = u"497070.0m/s"
	freq      = u"5.0THz"
	# emax      = u"0.5MV/cm"
	e = uconvert(u"C", 1u"eV" / 1u"V")
	ω = 2π*freq
	m = M * ħ * ω / 2

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", ζ * ħ * ω^2 / (2vf * e))
	us = scaledriving_frequency(freq, vf)

	t2 = Inf * u"s"
	t1 = Inf * u"s"

	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)

	axmax = df.eE / df.ω

	dt    = 0.1
	kxmax = 3axmax
	dkx   = 2kxmax / 1_000
	kymax = 1.0
	dky   = 1.0

	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_dt"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function make_dt_test(sim::Simulation,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dt,0.5)
	atolgoal = 1e-12
	rtolgoal = 1e-5
	maxtime = u"10minute"
	maxiter = 32 # min dt = 0.01/2^31 ≈ 5e-12
	return ConvergenceTest(sim,solver,method,atolgoal,rtolgoal,maxtime,maxiter;
		altpath = altpath)
end

const Ms = exp10.(LinRange(-1,1,32))
const ζs = exp10.(LinRange(-1,1,32))
const id = parse(Int,ENV["SLURM_ARRAY_TASK_ID"])
const ζ = ζs[id]

path(ζ) = joinpath(ENV["WORK"], "zeta-M-sweep/dt/zeta=$(ζ)")
altpath(ζ) = joinpath(ENV["HPCVAULT"], "zeta-M-sweep/dt/zeta=$(ζ)")
const sims = [make_simulation(m,ζ,path(ζ)) for m in Ms]
const tests = [make_dt_test(s,altpath(ζ)) for (s,m) in zip(sims,Ms)]

const logger = make_teelogger(
	joinpath(ENV["WORK"],"zeta-M-sweep/dt"), 
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

run!.(tests)

@info "$(now()): calculation finished."
