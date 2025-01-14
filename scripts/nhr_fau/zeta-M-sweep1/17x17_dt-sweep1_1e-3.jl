using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates

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


function makesim( M = 0.5, ζ = 4.0, dt = 0.1,
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

	axmax = df.eE / df.ω
	kxmax = 3axmax
	dkx   = kxmax / 500
	kymax = kxmax / 10
	dky   = kymax / 10

	pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_dt"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end

function maketest(M::Real,ζ::Real,dt_inter)
	subdir    = "zeta-M-sweep1/17x17_dt_1e-3"
	path      = joinpath(ENV["WORK"], subdir, "zeta=$(ζ)_M=$(M)")
    altpath   = joinpath(ENV["HPCVAULT"], subdir, "zeta=$(ζ)_M=$(M)")
    sim       = makesim(M,ζ,dt_inter(M,ζ),path,path)
    return maketest(sim,altpath)
end

function maketest(sim::Simulation,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dt,0.6)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-6,
		rtolgoal = 1e-3,
		maxtime = u"2*60minute",
		maxiterations = 12,
		path 	= joinpath(sim.datapath, "convergencetest_$(getname(method))_1e-4.hdf5"),
		altpath = altpath)
end

const cdat = DataFrame(CSV.File(joinpath(
    ENV["WORK"],
    "zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))

const rng         = [0.1,1.0,10.]
const dt_mat      = [cdat[cdat.M .== m .&& cdat.zeta .== ζ,:].dt |> first for m in rng,ζ in rng]
const dt_itp      = interpolate(dt_mat,BSpline(Cubic(Line(OnGrid()))))
const dt_sitp_log = scale(dt_itp, -1:1:1, -1:1:1)

dt_sitp(m,ζ) = dt_sitp_log(log10(m),log10(ζ))

# (m,ζ) coordinates for finer grid:
const coords_to_skip  = [(m,ζ) for m in rng for ζ in rng]
const coords          = filter(x -> x ∉ coords_to_skip,
    [(exp10(x),exp10(y)) for x in -1:0.125:1 for y in -1:0.125:1])
# 280 points in parameter space


const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const test 		= maketest(coords[id]...,dt_sitp) 

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_dt_1e-3"),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*"zeta=$(coords[id][1])_M=$(coords[id][2])"))

run!(test)

exit(0)
