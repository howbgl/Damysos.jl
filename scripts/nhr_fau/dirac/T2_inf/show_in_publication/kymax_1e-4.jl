using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates

import Damysos.getname
import Damysos.maximum_vecpot

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


global_logger(make_teelogger(
	joinpath(ENV["WORK"],"dirac/T2_inf/show_in_publication/kymax_1e-4"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"kymax_1e-4"))


const files = load_hdf5_files(joinpath(
	ENV["WORK"],
	"dirac/T2_inf/show_in_publication/dkx_1e-4"))

const oldsims = loadlast_testsim.(files)

const tests = [ConvergenceTest(f,LinearCUDA();
	resume = false,
	method = PowerLawTest(:kymax,1.4),
	rtolgoal = 1e-4,
	atolgoal = 1e-6,
	maxtime = u"12*60minute",
	maxiterations = 15,
	path = replace(f,"dky" => "kymax")) for f in files]


for t in tests
	run!(t)
end

exit(0)
