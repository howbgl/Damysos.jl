using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using Accessors

import Damysos.getname

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(),Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


function load_hdf5_files(path::String)
	allfiles = String[]
	for (root, dirs, files) in walkdir(path)
		for file in files
			file[end-4:end] == ".hdf5" && push!(allfiles,joinpath(root, file))
		end
	end
	@info "$(length(allfiles)) files found."
	return allfiles
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

	id = "M=$(M)_zeta=$(ζ)_dkx"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function makesim_interpol(ζ,M)
	npars  	= interpolate_pars(ζ,M)
	return makesim(
		ζ,
		M,
		0.003, # ensures rtol ≈ 0.05
		50*npars["dkx"],
		npars["dky"],
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
		atolgoal = 1e-8,
		rtolgoal = 0.1,
		maxtime = u"4minute",
		maxiterations = 10,
		path 	= path,
		altpath = altpath)
end


const coords = [(exp10(x),exp10(y)) for x in -1:0.125:1 for y in -1:0.125:1]
const dir 	 = joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_dkx_1e-1")
const dir2 	 = joinpath(ENV["HPCVAULT"],"zeta-M-sweep1/17x17_dkx_1e-1")
const tests_to_do = ConvergenceTest[]
const coords_to_do = []

	
for c in coords

	files = load_hdf5_files(joinpath(dir,"zeta=$(c[1])_M=$(c[2])"))
	if !isempty(files)
		if successful_retcode(files[1])
			continue
		else
			push!(tests_to_do,ConvergenceTest(
				files[1],
				resume = true,
				maxtime = u"6minute"))
			push!(coords_to_do,c)
		end
	else
		push!(tests_to_do,maketest(
			c[1],
			c[2],
			joinpath(dir,"zeta=$(ζ)_M=$(M)/dkxtest.hdf5"),
			joinpath(dir2,"zeta=$(ζ)_M=$(M)/dkxtest.hdf5")))
		push!(coords_to_do,c)
	end
end

for (t,c) in zip(tests_to_do,coords_to_do)
	
	global_logger(make_teelogger(dir,"zeta=$(c[1])_M=$(c[2])"))
	run!(t)
end

exit(0)
