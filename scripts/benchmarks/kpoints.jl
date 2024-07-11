using Damysos, TerminalLoggers, Dates, LoggingExtras, DataFrames, CSV, CUDA, CairoMakie
using BenchmarkTools

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(TerminalLogger(stderr,Logging.Info), info_logger, all_filelogger)
end

function runbench!(sims,fns,solver)
    alltimes = []
    for (s,f) in zip(sims,fns)
        # run once for compilation
        run!(s,f,solver;savedata=false,saveplots=false)

        times =  [@elapsed run!(s,f,solver;savedata=false,saveplots=false) for _ in 1:20]
        push!(alltimes,mean(times))
    end
    return alltimes
end

function make_demo_simulation(dkx = 1.0, datapath = "~/data", plotpath = datapath)

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

	dt    = 0.01
	kxmax = 175.0
	dky   = 1.0
	kymax = 10.0

	us   = scaledriving_frequency(freq, vf)
	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)
	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "kpoint_bench"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

const gpuname = CUDA.name(CUDA.device())
const path = joinpath(@__DIR__, "kpoint_on_$gpuname")
const altpath = joinpath(ENV["WORK"], "kpoint_on_$gpuname")
const sims = [make_demo_simulation(dkx,path) for dkx in exp10.(-0.6:-0.04:-1.5)]
const nks = [getnkx(s)*getnky(s) for s in sims]
const solver = LinearCUDA(10_000)
const fns = [define_functions(s,solver) for s in sims]

const logger = make_teelogger(path, "slurmid="*ENV["SLURM_JOB_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

times = runbench!(sims,fns,solver)

@show nks
@show times

data = DataFrame(:nk => nks,:runtime => times)
CSV.write(joinpath(path,"nk_vs_runtime_on_$(gpuname)_larger.csv"),data)

@info "$(now()): calculation finished."
