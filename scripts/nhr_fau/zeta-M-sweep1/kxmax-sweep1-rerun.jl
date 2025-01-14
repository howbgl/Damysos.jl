using Damysos, TerminalLoggers, Dates, LoggingExtras, DataFrames, CSV

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger         = MinLevelLogger(TerminalLogger(), Logging.Info)
	all_filelogger  = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

	return TeeLogger(tlogger, info_logger, all_filelogger)
end


const subdir    = "zeta-M-sweep1/dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2/zeta=0.1_M=10.0"
const path 		= joinpath(
	ENV["WORK"],
	subdir,
	"convergencetest_PowerLawTest_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.hdf5")
const newpath = replace(path,"kxmax_1e-2" => "kxmax_1e-2_rerun")

const test 		= ConvergenceTest(
	path, 
	LinearCUDA();
	resume = false,
	method = PowerLawTest(:kxmax,1.2),
	atolgoal = 1e-12,
	rtolgoal = 1e-2,
	maxtime = u"23*60minute",
	maxiterations = 40,
	path = path,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())),)


global_logger(make_teelogger(
	joinpath(ENV["WORK"], subdir),"slurmid=" * ENV["SLURM_JOB_ID"]))
@info "$(now())\nOn $(gethostname()):"
run!(test)

@info "$(now()): calculation finished."
