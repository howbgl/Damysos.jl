using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using CUDA, Accessors

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
	"dirac/T2_100fs/zeta-M-sweep1/3x3_dkx_1e-3"))

const oldsims = loadlast_testsim.(files)
const kymaxsims = [@set s.numericalparams.kymax = 0.2maximum_vecpot(s.drivingfield) for s in oldsims]
const sims  = [@set s.numericalparams.dky = s.numericalparams.kymax/50. for s in kymaxsims]
const newpaths = [replace(f,"dkx_1e-3" => "dky_1e-2") for f in files]
const newsims  = Simulation{Float64}[]

# Even though numerical convergence looks better, this data became worse for the smaller
# dkx, fix this manually:
for (s,f) in zip(sims,newpaths)
	if isnothing(match(r"zeta=0.1_M=10.0",f))
		push!(newsims,s)
	else
		push!(newsims,@set s.numericalparams.dkx = 0.012)
	end
end

const tests = [ConvergenceTest(s,LinearCUDA();
	resume = false,
	method = PowerLawTest(:dky,0.6),
	rtolgoal = 1e-2,
	atolgoal = 1e-7,
	maxtime = u"2*60minute",
	maxiterations = 12,
	path = replace(f,"dkx_1e-3" => "dky_1e-2")) for (s,f) in zip(newsims,newpaths)]


for t in tests
	run!(t)
	GC.gc(true)
	CUDA.reclaim()
end

exit(0)
