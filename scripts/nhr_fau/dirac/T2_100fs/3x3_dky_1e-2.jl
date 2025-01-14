using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates, Accessors

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
	joinpath(ENV["WORK"],"dirac/T2_100fs/zeta-M-sweep1/3x3_dky_1e-2"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"dky_1e-2"))


const files = load_hdf5_files(joinpath(
	ENV["WORK"],
	"dirac/T2_100fs/zeta-M-sweep1/3x3_kxmax_1e-3"))

const oldsims = loadlast_testsim.(files)
const kymaxsims = [@set s.numericalparams.kymax = 0.5maximum_vecpot(s.drivingfield) for s in oldsims]
const newsims  = [@set s.numericalparams.dky = 5.0 for s in kymaxsims]
const newpaths = [replace(f,"dkx" => "dky") for f in files]

const tests = [ConvergenceTest(s,LinearCUDA();
	resume = false,
	method = PowerLawTest(:dky,0.6),
	rtolgoal = 1e-2,
	atolgoal = 1e-6,
	maxtime = u"60minute",
	maxiterations = 12,
	path = replace(f,"kxmax_1e-3" => "dky_1e-2")) for (s,f) in zip(newsims,newpaths)]


for t in tests
	run!(t)
end

exit(0)
