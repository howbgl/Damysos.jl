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

const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "quadratic/zeta-M-sweep1/17x17_dt_1e-4"
const path 		= joinpath(ENV["WORK"], subdir)
const files 	= load_hdf5_files(path)
const tests 	= []

for f in files
	if !successful_retcode(f)
		push!(tests,ConvergenceTest(f,LinearCUDA();resume=true,maxtime=u"5*60minute"))
	end
end

@info "$(length(tests)) non-converged tests found."

global_logger(make_teelogger(
	joinpath(ENV["WORK"],subdir),
	"rerun1-slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

run!(tests[id])
exit(0)
