using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, Accessors

import Damysos.getname

function make_teelogger(logging_path::AbstractString, name::AbstractString)

	ensuredirpath(logging_path)
	info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
	info_logger     = MinLevelLogger(info_filelogger, Logging.Info)
	tlogger 		= MinLevelLogger(TerminalLogger(stdout),Logging.Info)
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

const sub  = "zeta-M-sweep1/dkx2_1e-4/zeta=10.0_M=0.1/convergencetest_PowerLawTest_dkx_1e-4.hdf5"
const subdir = "zeta-M-sweep1/dkx2_1e-4/rerun"
const path = joinpath(ENV["WORK"], sub)
const altpath 	= joinpath(ENV["HPCVAULT"],sub)

const testold 	= ConvergenceTest(
	path,
	LinearCUDA(10_000);
	resume = true)

const test = ConvergenceTest(
	testold.completedsims[end],
	LinearCUDA(12_000);
	method = PowerLawTest(:dkx,0.5),
	maxiterations = 32,
	maxtime = u"23*60minute",
	atolgoal = 1e-12,
	rtolgoal = 1e-4,
	path = joinpath(ENV["WORK"], subdir, basename(sub)))

global_logger(make_teelogger(joinpath(ENV["WORK"],subdir),"rerun_slurmid="*ENV["SLURM_JOB_ID"]))

@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)