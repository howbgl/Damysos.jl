using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using Accessors

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
	joinpath(ENV["WORK"],"dirac/T2_50fs/zeta-M-sweep1/3x3_dkx_1e-3"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"dkx_1e-3"))


const files = load_hdf5_files(joinpath(
	ENV["WORK"],
	"dirac/T2_50fs/zeta-M-sweep1/3x3_dt_1e-3"))

const sims = loadlast_testsim.(files)
const newsims = [@set s.numericalparams.dt = min(s.numericalparams.dt,0.01) for s in sims]
const tests = [ConvergenceTest(s,LinearCUDA(10_000);
	method = PowerLawTest(:dkx,0.6),
	resume = false,
	rtolgoal = 1e-3,
	atolgoal = 1e-6,
	maxtime = u"60minute",
	maxiterations = 14,
	path = replace(f,"dt" => "dkx")) for (s,f) in zip(newsims,files)]


for t in tests
	run!(t)
end

exit(0)
