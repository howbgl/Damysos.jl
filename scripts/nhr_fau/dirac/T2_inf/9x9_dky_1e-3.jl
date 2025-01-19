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

const dir  	 = joinpath(ENV["WORK"],"dirac/T2_inf/9x9_kxmax_1e-3")
const files  = load_hdf5_files(dir)
const id 	 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const file 	 = files[id]
const path 	 = replace(file,"kxmax_1e-3" => "dky_1e-3")


const kxmaxsim 	= Damysos.loadlast_testsim(file)
const kymax 	= 0.5maximum_vecpot(kxmaxsim.drivingfield)
const dky 		= kymax / 10 

# ζ=10 M=10 called nan_abort, but vx converged already in 1st iteration, fix that manually:
const oldpars = successful_retcode(file) ? kxmaxsim.numericalparams : @set kxmaxsim.numericalparams.kxmax = 320.0
const kymaxpars = @set oldpars.kymax = kymax
const pars 		= @set kymaxpars.dky = dky
const start 	= @set kxmaxsim.numericalparams = pars


const test   = ConvergenceTest(start,LinearCUDA(10_000);
	resume = false,
	method = PowerLawTest(:dky,0.6),
	rtolgoal = 1e-3,
	atolgoal = 1e-7,
	path = path,
	maxtime = u"4*60minute",
	maxiterations = 15)
	

global_logger(make_teelogger(dirname(path),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*first(splitext(basename(file)))))

run!(test)

exit(0)
