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

function extract_parameters(f)
	z,m = parse.(Float64,match(r"zeta=([0-9.]+)_M=([0-9.]+)",f).captures)
	return (ζ=z,M=m)
end

function maketest(sim::Simulation,path::String,altpath::String=pwd())

	cdat = DataFrame(CSV.File(joinpath(
		ENV["WORK"],
    	"zeta-M-sweep1/9x9_conv_dt_1e-4_dkx_1e-2_dky_1e-2_kxmax_1e-2.csv")))
	
	rng         	= [0.1,1.0,10.]
	kymax_path 		= [cdat[cdat.M .== m .&& cdat.zeta .== ζ,:].kymax |> first for m in rng,ζ in rng]
	kymax_itp    	= interpolate(kymax_path,BSpline(Linear()))
	kymax_sitp_log 	= scale(kymax_itp, -1:1:1, -1:1:1)

	kymax_sitp(m,ζ) = kymax_sitp_log(log10(m),log10(ζ))

	pars = extract_parameters(path)
	s  	 = @set sim.numericalparams.kymax = kymax_sitp(pars.M,pars.ζ)

	# make k-grid rhougher in y-direction if nky > 1_000
	dky 	= getparams(s).nky < 1_000 ? s.numericalparams.dky : 2s.numericalparams.kymax / 1_000
	newsim 	= @set s.numericalparams.dky = dky
	solver 	= LinearCUDA(10_000)
	method 	= PowerLawTest(:kymax,1.3)

	return ConvergenceTest(newsim,solver;
		method = method,
		atolgoal = 1e-6,
		rtolgoal = 1e-2,
		maxtime = u"5.9*60minute",
		maxiterations = 42,
		path 	= path,
		altpath = altpath)
end

const files_to_do 	= DataFrame(CSV.File(
	"/home/hpc/b228da/b228da10/Damysos.jl/scripts/nhr_fau/kymax_files_to_do.csv"))
const id 			= parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
const path 			= files_to_do.filepath[id]
const pars 			= extract_parameters(path)
const oldsim 		= Damysos.loadlast_testsim(path)
const test 			= maketest(oldsim,replace(path,"kxmax_1e-2" => "kymax_1e-2")) 

global_logger(make_teelogger(
	joinpath(ENV["WORK"],"zeta-M-sweep1/17x17_kymax_1e-2"),
	"slurmid="*ENV["SLURM_ARRAY_TASK_ID"]*"zeta=$(pars.ζ)_M=$(pars.M)"))

run!(test)

exit(0)
