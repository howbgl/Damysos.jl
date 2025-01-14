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

const id 		 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2_dky_1e-2_kxmax_1e-3"
const files		= load_hdf5_files(joinpath(ENV["WORK"],"quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2_dky_1e-2"))
const path 		= files[id]
const m 		= match(r"zeta=([0-9.]+)_M=([0-9.]+)",path)
const newpath 	= joinpath(
	ENV["WORK"],
	subdir,
	m.match,
	"convergencetest_LinearTest_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-3.hdf5")

const test 		= ConvergenceTest(
	path,
	LinearCUDA();
	resume = false,
	method = PowerLawTest(:kxmax,1.2),
	atolgoal = 1e-12,
	rtolgoal = 1e-3,
	maxtime = u"10*60minute",
	maxiterations = 20,
	path 	= newpath,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))


global_logger(make_teelogger(joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)