using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, HDF5

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
const subdir 	= "quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2_dky_1e-2"
const files		= load_hdf5_files(joinpath(ENV["WORK"],"quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2"))
const path 		= files[id]
const newpath 	= replace(path,"dt_1e-6_dkx_1e-2" => "dt_1e-6_dkx_1e-2_dky_1e-2")
const test 		= ConvergenceTest(
	path,
	LinearCUDA();
	resume = false,
	method = PowerLawTest(:dky,0.65),
	atolgoal = 1e-12,
	rtolgoal = 1e-2,
	maxtime = u"30minute",
	maxiterations = 40,
	path 	= newpath,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))

const logger = make_teelogger(
	joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)