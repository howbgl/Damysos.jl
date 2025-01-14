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


const files = load_hdf5_files(joinpath(ENV["WORK"],"dirac/T2_inf/converge_r_9x9_dkx"))
const id 	 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const test 	 = ConvergenceTest(files[id],LinearCUDA(10_000);
	resume = true,
	maxiterations=4)

	
global_logger(make_teelogger(joinpath(ENV["WORK"],"dirac/T2_inf/converge_r_9x9_dkx"),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*basename(files[id])[1:end-5]))

run!(test)

exit(0)
