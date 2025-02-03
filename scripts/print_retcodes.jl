using Damysos, CairoMakie, ArgParse, HDF5, DataFrames, CSV, ProgressLogging, TerminalLoggers

import Damysos.plotdata

function parse_cmdargs()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"rootpath"
            help = "Root path containing all .hdf5 files to be loaded"
            arg_type = String
            required = true
	end
	return parse_args(s)
end


function load_hdf5_files(path::String)
	allfiles = String[]
	for (root, dirs, files) in walkdir(path)
		for file in files
			file[end-4:end] == ".hdf5"  && push!(allfiles,joinpath(root, file))
		end
	end
	@info "$(length(allfiles)) files found."
	return allfiles
end

Base.global_logger(TerminalLogger())

cmdargs = parse_cmdargs()
files 	= load_hdf5_files(cmdargs["rootpath"])
codes 	= []

for f in files
	h5open(f,"r") do file
		if "testresult" ∈ keys(file)
			push!(codes,ReturnCode.T(read(file["testresult"],"retcode")))
		else
			push!(codes,ReturnCode.running)
		end
	end
end

for (f,c) in zip(files,codes)
	println(f * ":	" * string(c))
end

println("\n\n# successes = $(sum(successful_retcode.(codes)))")
println("# fails = $(sum((!).(successful_retcode.(codes))))")
