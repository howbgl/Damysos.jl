using Damysos, TerminalLoggers, Dates, LoggingExtras, CSV, HDF5, Accessors, CUDA

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


const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const subdir 	= "zeta-M-sweep1/t2-sweep-kymax-rough"
const t2s 		= [0.1,0.25,0.5,0.75,1.0,2.0]

global_logger(make_teelogger(
	joinpath(ENV["WORK"],subdir),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]))


files 	= load_hdf5_files(joinpath(ENV["WORK"],"zeta-M-sweep1/kymax-rough"))

const path 		= files[id]
const m 		= match(r"zeta=([0-9.]+)_M=([0-9.]+)",path)
const newpath 	= joinpath(ENV["WORK"],subdir,m.match)
const oldsim = Damysos.loadlast_testsim(path)
const h 	 = oldsim.liouvillian.hamiltonian
const newLs  = [TwoBandDephasingLiouvillian(h, 0.0, t2) for t2 in t2s]
const sims 	 = [Simulation(
	l, 
	oldsim.drivingfield,
	oldsim.numericalparams,
	oldsim.observables, 
	oldsim.unitscaling,
	m.match*"_t2=$(l.t2)",
	joinpath(newpath,"t2=$(l.t2)"), 
	joinpath(newpath,"t2=$(l.t2)")) for l in newLs]
const solver = LinearCUDA(10_000,GPUVern7(),1)
const fns 	 = [define_functions(s,solver) for s in sims]

for (s,fs) in zip(sims,fns)
	GC.gc(true)
	CUDA.reclaim()
	run!(s,fs,solver)
end

@info "$(now()): calculation finished."

exit(res.retcode)