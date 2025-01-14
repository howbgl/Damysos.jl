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
		0.005,
		10npars["dkx"],
		10npars["dky"],
		npars["kxmax"],
		npars["kymax"])
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
	method = PowerLawTest(:dt,0.6)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-7,
		rtolgoal = 1e-7,
		maxtime = u"20minute",
		maxiterations = 12,
		path 	= path,
		altpath = altpath)
end

const coords = [
	(3.16,0.177),
	(3.16,3.16),
	(3.16,7.50),
	(0.56,0.177),
	(1.78,0.177),
	(7.50,0.177)]

const dir 		= joinpath(ENV["WORK"],"dirac/T2_inf/show_in_publication/dt_1e-7")
const paths 	= [joinpath(dir,"zeta=$(c[1])_M=$(c[2])") for c in coords]
const tests		= [maketest(c[1],c[2],
	joinpath(p,"dt_1e-7.hdf5")) for (c,p) in zip(coords,paths)] 

global_logger(make_teelogger(dir,"slurmid="*ENV["SLURM_JOB_ID"]*"dt_1e-7"))

for t in tests
	run!(t)
end

exit(0)
