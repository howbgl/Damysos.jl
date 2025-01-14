using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates

import Damysos.getname
import Damysos.maximum_vecpot

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end

function makesim(i=1;datapath = pwd(), plotpath = datapath)

	cdat = DataFrame(CSV.File(joinpath(
		ENV["WORK"],
		"zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))
	
	pars_row  = cdat[i,:]
	ζ = pars_row.zeta
	M = pars_row.M
    dkx = 8pars_row.dkx
	dky = pars_row.dky
	dt = 0.05

	subdir    = "dirac/T2_100fs/zeta-M-sweep1/3x3_dt_1e-3"		
	path = joinpath(ENV["WORK"], subdir, "zeta=$(pars_row.zeta)_M=$(pars_row.M)")

	vf = u"497070.0m/s"
	freq = u"5.0THz"
	# emax      = u"0.5MV/cm"
	ω = 2π * freq
	m = M * ħ * ω / 2

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", ζ * ħ * ω^2 / (2vf * q_e))
	us = scaledriving_frequency(freq, vf)

	t2 = u"100.0fs"
	t1 = Inf * u"s"

	h  = GappedDirac(energyscaled(m, us))
	l  = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df = GaussianAPulse(us, σ, freq, emax)

	kxmax 	= 3maximum_vecpot(df)
	kymax 	= 0.0
	pars  	= NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs   	= [Velocity(pars), Occupation(pars)]
	id 		= "M=$(M)_zeta=$(ζ)_dt"

	return Simulation(l, df, pars, obs, us, id, path, path)
end

function maketest(sim::Simulation;altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dt,0.6)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-6,
		rtolgoal = 1e-3,
		maxtime = u"10minute",
		maxiterations = 12,
		path 	= joinpath(sim.datapath, "convergencetest_$(getname(method))_1e-2.hdf5"),
		altpath = altpath)
end


const sims      = makesim.(1:9)
const tests		= maketest.(sims) 

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"dirac/T2_100fs/zeta-M-sweep1/3x3_dt_1e-3"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"dt_1e-3"))

for t in tests
	run!(t)
end

exit(0)
