using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using Accessors

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

const id 			= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const kxmax_dir		= joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_kxmax_1e-2")
const kxmax_files	= load_hdf5_files(kxmax_dir)
const failed_paths 	= kxmax_files[(!).(successful_retcode.(kxmax_files))]

length(failed_paths) != 6 && throw(ErrorException("length(failed_paths) != 6"))

const capt 			= match(r"zeta=([0-9.]+)_M=([0-9.]+)",failed_paths[id])
const ζ 			= parse(Float64,capt[1])
const M 			= parse(Float64,capt[2])

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_kxmax_1e-2"),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*"zeta=$(ζ)_M=$(M)"))

const test = ConvergenceTest(
	failed_paths[id],
	LinearCUDA();
	maxtime = u"23.7*60minute",
	resume = true)

run!(test)

exit(0)
