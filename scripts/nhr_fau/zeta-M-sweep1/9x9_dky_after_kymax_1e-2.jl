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


const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const paths 	= load_hdf5_files(joinpath(
	ENV["WORK"],
	"zeta-M-sweep1/9x9_kymax_first_1e-2"))
const oldpath 	= paths[id]
const test 		= ConvergenceTest(
	oldpath,
	LinearCUDA();
	method = PowerLawTest(:dky,0.6),
	maxtime=u"23*60minute",
	maxiterations=12,
	path = replace(oldpath,"kymax_first" => "dky_after_kymax"))

global_logger(
	make_teelogger(dirname(test.testdatafile),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

run!(test)

exit(0)
