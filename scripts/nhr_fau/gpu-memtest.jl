using Damysos, TerminalLoggers, Dates, LoggingExtras, DataFrames, CSV, BenchmarkTools


function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(TerminalLogger(stderr,Logging.Info), info_logger, all_filelogger)
end

function make_test_simulation1(
	dt::Real = 0.01,
	dkx::Real = 1.0,
	dky::Real = 1.0,
	kxmax::Real = 175,
	kymax::Real = 100;
    datapath = ENV["WORK"],
    plotpath = ENV["WORK"])

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

	id    = "gpumemtest1"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

const logger = make_teelogger(
	joinpath(ENV["WORK"], "gpu-memtest"),
	"slurmid=" * ENV["SLURM_JOB_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

const path = joinpath(ENV["WORK"], "gpu-memtest")
const altpath = joinpath(ENV["HPCVAULT"], "gpu-memtest")
const dts = exp10.(-2:-0.5:-4)
const sims = [make_test_simulation1(dt;datapath=path,plotpath=path) for dt in dts]
const solver = LinearCUDA(10_000)
const fns = [define_functions(s,solver) for s in sims]
const runtimes = [@elapsed run!(s,fs,solver;savedata=false,saveplots=false) for (s,fs) in zip(sims,fns)]

dat = DataFrame(:nt => getnt.(sims),:runtime => runtimes)
CSV.write(joinpath(path,"runtimes.csv"),dat)