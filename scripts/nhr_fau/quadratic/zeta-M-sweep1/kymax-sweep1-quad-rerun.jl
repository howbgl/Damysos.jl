using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, HDF5

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

const id 		 = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2_dky_1e-2_kxmax_1e-2_kymax_1e-2_atol_1e-6"
# const files		= load_hdf5_files(joinpath(ENV["WORK"],"quadratic/zeta-M-sweep1/dt_1e-6_dkx_1e-2_dky_1e-2_kxmax_1e-2"))

const files 			= load_hdf5_files(joinpath(ENV["WORK"],subdir))
const cfiles 			= String[]

for f in files
	if !Damysos.successful_retcode(f) 
		push!(cfiles,f)
	end
end

const path 	 = cfiles[id]
const oldsim = Damysos.loadlast_testsim(path)
const kymax  = oldsim.numericalparams.kymax
	
const test 		= ConvergenceTest(
	path,
	LinearCUDA(10_000,GPUVern7(),4);
	resume = true,
	method = LinearTest(:kymax,0.4kymax),
	atolgoal = 1e-6,
	rtolgoal = 1e-2,
	maxtime = u"23*60minute",
	maxiterations = 80,
	path 	= path,
	altpath = joinpath(ENV["HPCVAULT"], subdir, basename(tempname())))


global_logger(make_teelogger(joinpath(ENV["WORK"],subdir),"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))

@info "$(now())\nOn $(gethostname()):"

res = run!(test)

@info "$(now()): calculation finished."

exit(res.retcode)