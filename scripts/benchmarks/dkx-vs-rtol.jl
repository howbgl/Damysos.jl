using Damysos, TerminalLoggers, Dates, LoggingExtras, DataFrames, CSV, CUDA, CairoMakie
using BenchmarkTools

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensurepath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(TerminalLogger(stderr,Logging.Info), info_logger, all_filelogger)
end

function make_simulation(
	M = 0.5,
	ζ = 4.0,
	dt =0.01,
	dkx = 0.1;
	datapath = "~/data",
	plotpath = datapath)

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

	ky 	  = 0.0
	kxmax = 3df.eE / df.ω

	pars = NumericalParams1d(dkx, kxmax, ky, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_dkx_vs_rtol"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

const gpuname = CUDA.name(CUDA.device())
const path = joinpath(@__DIR__, "dkx_vs_rtol_on_$gpuname")
const altpath = joinpath(pwd(), "dkx_vs_rtol_on_$gpuname")

const parpairs = [(z,m) for m in [0.1,1.,10.] for z in [0.1,1.,10.]]
const sims = [make_simulation(m,z,2e-5,5.0;datapath=path) for (z,m) in parpairs]
const solver = LinearCUDA(10_000)
const rtols = exp10.(-1:-0.2:-8)
const m = PowerLawTest(:dkx,0.5)
const ctests = [
	ConvergenceTest(s,solver,m,1e-12,rtol,u"60minute",16) for rtol in rtols for s in sims]

const logger = make_teelogger(path,"dkx-vs-rtol")

# global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = [run!(ct;savedata=false) for ct in ctests]
times = [x.retcode == 0 ? x.elapsed_time_sec : missing for x in res]

@show res
@show times

data = DataFrame(
	:rtol => rtols,
	:runtime => times,
	:zeta => [x[1] for x in parpairs],
	:m => [x[2] for x in parpairs])

CSV.write(joinpath(path,"rtol_vs_runtime_on_$gpuname.csv"),data)

@info "$(now()): calculation finished."
