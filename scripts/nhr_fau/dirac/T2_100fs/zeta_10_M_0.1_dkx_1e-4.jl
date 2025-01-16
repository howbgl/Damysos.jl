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

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"dirac/T2_100fs/zeta-M-sweep1/3x3_dkx_1e-3"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"zeta=10.0_M=0.1_dkx_1e-3"))

const path = joinpath(
	ENV["WORK"],
	"dirac/T2_100fs/zeta-M-sweep1/3x3_dkx_1e-3/zeta=0.1_M=10.0/convergencetest_PowerLawTest_dkx_1e-3.hdf5")
const test = ConvergenceTest(
	path,
	LinearCUDA();
	rtolgoal = 1e-4,
	resume = true,
	maxiterations = 12,
	maxtime = u"4*60minute")


run!(test)

exit(0)
