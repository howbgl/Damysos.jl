using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates
using Accessors

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


function maketest(sim::Simulation,path::String,altpath::String=pwd())
	solver = LinearCUDA(10_000)
	method = PowerLawTest(:dky,0.6)
	return ConvergenceTest(sim,solver;
		method = method,
		atolgoal = 1e-6,
		rtolgoal = 1e-2,
		maxtime = u"4*60minute",
		maxiterations = 12,
		path 	= path,
		altpath = altpath)
end

const id 		= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const dkx_dir 	= joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_dkx_1e-2")
const dkx_path 	= load_hdf5_files(dkx_dir)[id]
const oldsim 	= Damysos.loadlast_testsim(dkx_path)

const cdat = DataFrame(CSV.File(joinpath(
	ENV["WORK"],
    "zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))
	
const rng         	= [0.1,1.0,10.]
const dky_mat 		= [cdat[cdat.M .== m .&& cdat.zeta .== ζ,:].dky |> first for m in rng,ζ in rng]
const dky_itp      	= interpolate(dky_mat,BSpline(Linear()))
const dky_sitp_log 	= scale(dky_itp, -1:1:1, -1:1:1)

dky_sitp(m,ζ) = dky_sitp_log(log10(m),log10(ζ))

const capt 		= match(r"zeta=([0-9.]+)_M=([0-9.]+)",dkx_path)
const ζ 		= parse(Float64,capt[1])
const M 		= parse(Float64,capt[2])
const sim 		= @set oldsim.numericalparams.dky = dky_sitp(M,ζ)

const test 		= maketest(sim,replace(dkx_path,"dkx_1e-2" => "dky_1e-2","dt_1e-4" => "dky_1e-2")) 

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_dky_1e-2"),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*"zeta=$(ζ)_M=$(M)"))

run!(test)

exit(0)
