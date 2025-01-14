using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, HDF5

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

files = [
	"quadratic/zeta-M-sweep1/dky2_1e-4/zeta=1.0_M=0.1/convergencetest_PowerLawTest_dky_1e-4.hdf5",
	"quadratic/zeta-M-sweep1/dky2_1e-4/zeta=1.0_M=1.0/convergencetest_PowerLawTest_dky_1e-4.hdf5",
	"quadratic/zeta-M-sweep1/dky2_1e-4/zeta=10.0_M=0.1/convergencetest_PowerLawTest_dky_1e-4.hdf5",
	"quadratic/zeta-M-sweep1/dky2_1e-4/zeta=10.0_M=1.0/convergencetest_PowerLawTest_dky_1e-4.hdf5",
	"quadratic/zeta-M-sweep1/dky2_1e-4/zeta=10.0_M=10.0/convergencetest_PowerLawTest_dky_1e-4.hdf5"]

files = [joinpath(ENV["WORK"],f) for f in files]

const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "quadratic/zeta-M-sweep1/dky2_1e-4"
const path 		= files[id]
const test 		= ConvergenceTest(
	path,
	LinearCUDA(10_000);
	resume = true)

const logger = make_teelogger(
	joinpath(ENV["WORK"],subdir),"rerun_slurmid="*ENV["SLURM_ARRAY_TASK_ID"])

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)