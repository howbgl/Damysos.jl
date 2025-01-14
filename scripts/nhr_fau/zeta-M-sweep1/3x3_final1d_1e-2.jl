using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates

import Damysos.getname

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(stdout),Logging.Info)
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


function makesim( M = 0.5, ζ = 4.0, dt = 0.1, dkx = 1.0, kxmax = 100,
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

	pars = NumericalParams1d(dkx, kxmax, 0.0, dt, -5df.σ)
	obs  = [Velocity(pars), Occupation(pars)]

	id = "M=$(M)_zeta=$(ζ)_final1d_1e-2"

	return Simulation(l, df, pars, obs, us, id, datapath, plotpath)
end


const cdat = DataFrame(CSV.File(joinpath(
    ENV["WORK"],
    "zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))


const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const pars_row  = cdat[id,:]
const subdir    = "zeta-M-sweep1/3x3_final1d_1e-2"
const path      = joinpath(ENV["WORK"], subdir, "zeta=$(pars_row.zeta)_M=$(pars_row.M)")
const altpath   = joinpath(ENV["HPCVAULT"], subdir, "zeta=$(pars_row.zeta)_M=$(pars_row.M)")
const sim       = makesim(
    pars_row.M,
    pars_row.zeta,
    pars_row.dt,
    pars_row.dkx,
    pars_row.kxmax,
	path)


global_logger(make_teelogger(
	joinpath(ENV["WORK"],subdir),
	"slurmid=$(id)_zeta=$(pars_row.zeta)_M=$(pars_row.M)"))

const solver = LinearCUDA()
const fns 	 = define_functions(sim,solver)

run!(sim,fns,solver;savedata=true,saveplots=true)

exit(0)
