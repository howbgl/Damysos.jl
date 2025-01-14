using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV

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
const subdir 	= "zeta-M-sweep1/dkx2_1e-4"
files 			= load_hdf5_files(joinpath(ENV["WORK"],"zeta-M-sweep1/dt1"))
const path 		= files[id]
const newpath 	= replace(path,"dt" => "dkx", "1e-6" => "1e-4")
const test 	= ConvergenceTest(
	path,
	LinearCUDA(10_000);
	resume = false,
	method = PowerLawTest(:dkx,0.5),
	atolgoal = 1e-12,
	rtolgoal = 1e-4,
	maxtime = u"10*60minute",
	maxiterations = 40,
	path 	= newpath,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))


global_logger(make_teelogger(joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)