using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using Accessors

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



function makesim( ζ = 4.0,M = 0.5, dt = 0.1, dkx= 1., dky = 1., kxmax = 100., kymax = 10.,
	datapath = "~/data", 
	plotpath = datapath)

	vf = u"497070.0m/s"
	freq = u"5.0THz"
	# emax      = u"0.5MV/cm"
	ω = 2π * freq
	m = M * ħ * ω / 2

	σ = uconvert(u"fs", 1.5 / freq)
	emax = uconvert(u"MV/cm", ζ * ħ * ω^2 / (2vf * q_e))
	us = scaledriving_frequency(freq, vf)

	t2 = Inf * u"s"
	t1 = Inf * u"s"

	h  = GappedDirac(energyscaled(m, us))
	l  = TwoBandDephasingLiouvillian(h, timescaled(t1, us), timescaled(t2, us))
	df = GaussianAPulse(us, σ, freq, emax)
	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars)]

	id = "M=$(M)_zeta=$(ζ)_dt"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function makesim_interpol(ζ,M)
	npars  	= interpolate_pars(ζ,M)
	return makesim(
		ζ,
		M,
		0.003,
		1.0,
		10npars["dky"],
		npars["kxmax"],
		0.0)
end

function interpolate_pars(zeta,multiphoton)
	cdat = DataFrame(CSV.File(
		joinpath(
            ENV["WORK"],
            "zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))

	pars = Dict()
	rng = [0.1,1.0,10.]
	for par in ["dt","dkx","dky","kxmax","kymax"]
		mat = [cdat[cdat.M .== m .&& cdat.zeta .== ζ,:][!,par] |> first for m in rng,ζ in rng]
		itp = interpolate(mat,BSpline(Linear()))

		sitp_log 	= scale(itp, -1:1:1, -1:1:1)
		sitp(m,z) 	= sitp_log(log10(m),log10(z))
		pars[par] 	= sitp(multiphoton,zeta)
	end
	return pars
end


function maketest(ζ,M,path::String,altpath::String=pwd())
	sim    = makesim_interpol(ζ,M)
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dkx,0.6)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-7,
		rtolgoal = 1e-4,
		maxtime = u"60minute",
		maxiterations = 14,
		path 	= path,
		altpath = altpath)
end


const coords = [(exp10(x),exp10(y)) for x in -1:0.25:1 for y in -1:0.25:1]
const dir 	 = joinpath(ENV["WORK"],"dirac/T2_inf/9x9_dkx_1e-4")
const dir2 	 = joinpath(ENV["HPCVAULT"],"dirac/T2_inf/9x9_dkx_1e-4")
const id 	 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const c 	 = coords[id]
const test   = maketest(
	c[1],
	c[2],
	joinpath(dir,"zeta=$(c[1])_M=$(c[2]).hdf5"),
	joinpath(dir2,"zeta=$(c[2])_M=$(c[2]).hdf5"))

	
global_logger(make_teelogger(dir,
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*"zeta=$(c[1])_M=$(c[2])"))

run!(test)

exit(0)
