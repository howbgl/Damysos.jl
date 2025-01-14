using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, HDF5, Accessors

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

const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "zeta-M-sweep1/kymax-rough"

global_logger(make_teelogger(
	joinpath(ENV["WORK"],subdir),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

files 	= load_hdf5_files(
	joinpath(ENV["WORK"],"zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-3"))

const path 		= files[id]
const m 		= match(r"zeta=([0-9.]+)_M=([0-9.]+)",path)
const newpath 	= joinpath(
	ENV["WORK"],
	subdir,
	m.match,
	"convergencetest_LinearTest_dt_1e-4_dkx_1e-2_kxmax_1e-2_kymax_1e-2_atol_1e-6.hdf5")
const oldsim = Damysos.loadlast_testsim(path)
const oldpars = oldsim.numericalparams
const olddkx = oldpars.dkx
const newpars = @set oldpars.dky = 2olddkx
const newsim = @set oldsim.numericalparams = newpars

const test 	= ConvergenceTest(
	newsim,
	LinearCUDA(8_000,GPUVern7(),1);
	resume = false,
	method = PowerLawTest(:kymax,1.15),
	atolgoal = 1e-6,
	rtolgoal = 1e-2,
	maxtime = u"23*60minute",
	maxiterations = 128,
	path 	= newpath,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))



res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)