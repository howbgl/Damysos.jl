using Damysos, CSV, DataFrames, Interpolations, HDF5, TerminalLoggers, LoggingExtras, Dates

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


global_logger(make_teelogger(
	joinpath(ENV["WORK"],"dirac/T2_inf/show_in_publication/dky_1e-4"),
	"slurmid="*ENV["SLURM_JOB_ID"]*"dky_1e-4_continue"))


const path = joinpath(
	ENV["WORK"],
	"dirac/T2_inf/show_in_publication/dky_1e-4/zeta=7.5_M=0.177/dky_1e-4.hdf5")

const test = ConvergenceTest(path,LinearCUDA();
	resume = true,
	maxtime = u"4*60minute",
	maxiterations = 12) 

run!(test)

exit(0)
