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
	joinpath(ENV["WORK"],"dirac/T2_100fs/zeta-M-sweep1/3x3_kxmax_1e-3"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"kxmax_1e-3"))


const files = load_hdf5_files(joinpath(
	ENV["WORK"],
	"dirac/T2_100fs/zeta-M-sweep1/3x3_dkx_1e-2"))

const tests = [ConvergenceTest(f,LinearCUDA();
	resume = false,
	method = PowerLawTest(:kxmax,1.3),
	rtolgoal = 1e-3,
	atolgoal = 1e-6,
	maxtime = u"60minute",
	maxiterations = 20,
	path = replace(f,"dkx" => "kxmax", "1e-2" => "1e-3")) for f in files]


for t in tests
	run!(t)
end

exit(0)
