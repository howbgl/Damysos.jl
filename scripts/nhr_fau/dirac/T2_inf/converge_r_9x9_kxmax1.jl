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

const dir 	 = joinpath(ENV["WORK"],"dirac/T2_inf/converge_r_9x9_dky1")
const files  = load_hdf5_files(dir)
const id 	 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const file 	 = files[id]
const path 	 = replace(file,"dky1" => "kxmax1")
const test   = ConvergenceTest(file,LinearCUDA(10_000);
	resume = false,
	method = PowerLawTest(:kxmax,1.2),
	path = path,
	maxtime = u"5.5*60minute",
	maxiterations = 4)
	
global_logger(make_teelogger(dir,
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*basename(file)[1:end-5]))

run!(test)

exit(0)
