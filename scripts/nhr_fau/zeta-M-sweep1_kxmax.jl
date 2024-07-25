using Damysos, TerminalLoggers, Dates, LoggingExtras, DataFrames, CSV

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger         = MinLevelLogger(TerminalLogger(), Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function make_simulation( M = 0.5, ζ = 4.0, dt =0.01, dkx = 0.1, datapath = "~/data", plotpath = datapath)

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

	kxmax = 3axmax
	ky 	  = 0.0

	pars = NumericalParams1d(dkx, kxmax, ky, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_kxmax"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function make_kxmax_test(sim::Simulation, altpath::String = pwd())
	solver = LinearCUDA(10_000)
	shift  = 0.4 * sim.numericalparams.kxmax
	method = LinearTest(:kxmax, shift)
	atolgoal = 1e-12
	rtolgoal = 1e-5
	maxtime = u"8minute"
	maxiter = 32
	return ConvergenceTest(sim, solver, method, atolgoal, rtolgoal, maxtime, maxiter;
		altpath = altpath)
end

const id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const start = CSV.File(joinpath(
	ENV["HOME"],
	"Damysos.jl/scripts/nhr_fau/zeta-M-sweep1-conv-dkx.csv"))

const zetas = Damysos.subdivide_vector(start.zeta,32)[id]
const ms 	= Damysos.subdivide_vector(start.M,32)[id]
const dkxs 	= Damysos.subdivide_vector(start.dkx,32)[id]
const dts 	= Damysos.subdivide_vector(start.dt,32)[id]


path(ζ) = joinpath(ENV["WORK"], "zeta-M-sweep/1d/kxmax/zeta=$(ζ)")
altpath(ζ) = joinpath(ENV["HPCVAULT"], "zeta-M-sweep/1d/kxmax/zeta=$(ζ)")
const sims = [make_simulation(m, ζ, dt, dkx, path(ζ)) for (ζ,m,dt,dkx) in zip(zetas,ms,dts,dkxs)]
const tests = [make_kxmax_test(s, altpath(ζ)) for (s, ζ) in zip(sims, zetas)]

const logger = make_teelogger(
	joinpath(ENV["WORK"], "zeta-M-sweep/1d/kxmax"),
	"slurmid=" * ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

for t in tests
	run!(t)
	GC.gc(true)
end

@info "$(now()): calculation finished."
