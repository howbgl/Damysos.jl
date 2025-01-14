using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, Accessors

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
const subdir 	= "quadratic/zeta-M-sweep1/new_dt_1e-6_dkx_1e-3"
files 			= load_hdf5_files(joinpath(ENV["WORK"],"quadratic/zeta-M-sweep1/dt1"))
const path 		= files[id]
const newpath 	= replace(path,"dt1" => "new_dt_1e-4_dkx_1e-3", "dt_1e-6" => "new_dt_1e-6_dkx_1e-3")
const oldsim 	= Damysos.loadlast_testsim(path)
const oldpars 	= oldsim.numericalparams
const newpars 	= @set oldpars.dkx = 10oldpars.dkx
const newsim 	= @set oldsim.numericalparams = newpars

const test 	= ConvergenceTest(
	newsim,
	LinearCUDA(10_000);
	resume = false,
	method = PowerLawTest(:dkx,0.7),
	atolgoal = 1e-12,
	rtolgoal = 1e-3,
	maxtime = u"30minute",
	maxiterations = 30,
	path 	= newpath,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))

const logger = make_teelogger(
	joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)