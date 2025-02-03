using Damysos, CairoMakie, ArgParse, HDF5, DataFrames, CSV, ProgressLogging, TerminalLoggers

import Damysos.plotdata

function parse_cmdargs()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"rootpath"
            help = "Root path containing all .hdf5 files to be loaded"
            arg_type = String
            required = true
		"--maxgraphs", "-m"
			help = "Maximum number of different graphs (with different parameters) in plot"
			arg_type = Int
			default = 10
	end
	return parse_args(s)
end

"""
Finds all files with a given extension in a directory and its subdirectories.

# Arguments
- `dir_path::String`: The path to the directory.
- `extension::String`: The desired file extension (e.g., "txt", "jpg").

# Returns
- `Vector{String}`: A vector containing paths to all matching files.
"""
function find_files_with_extension(dir_path::String, extension::String)::Vector{String}
    # Ensure the extension starts with a dot
    ext = startswith(extension, ".") ? extension : "." * extension

    # Container to store matching files
    matching_files = String[]

    # Recursive helper function to search directories
    function search_directory(current_dir::String)
        for entry in readdir(current_dir, join=true)
            if isfile(entry) && endswith(entry, ext)
                push!(matching_files, entry)
            elseif isdir(entry)
                search_directory(entry)
            end
        end
    end

    # Start the search from the given directory
    search_directory(dir_path)

    return matching_files
end


function load_hdf5_files(path::String)
	allfiles = String[]
	for (root, dirs, files) in walkdir(path)
		for file in files
			file[end-4:end] == ".hdf5" && file[1:4] == "conv" && push!(allfiles,
					joinpath(root, file))
		end
	end
	@info "$(length(allfiles)) files found."
	return allfiles
end

function plotsims(path::String,maxgraphs_per_plot::Integer=10,title="")

	h5open(path,"r") do file
		try
			retcode = "testresult" ∈ keys(file) ? string(ReturnCode.T(read(file["testresult"],"retcode"))) : "retcode not found"
			method 	= "method" ∈ keys(file) ? Damysos.getname(Damysos.load_obj_hdf5(file["method"])) : "method not found"
			g 		= file["completedsims"]
			sims 	= [Damysos.load_obj_hdf5(g[s]) for s in keys(g)]

			sort!(sims,by=Damysos.getsimindex)

			sims  = length(sims) > maxgraphs_per_plot ? sims[end-maxgraphs_per_plot:end] : sims
			title = isnothing(title) ? Damysos.stringexpand_vector([s.id for s in sims]) : title
			sims = filter(s -> 0 == Damysos.count_nans(s.observables),sims)
			@show sims[1].observables
			if isempty(sims)
				@warn "No simulations to plot!"
				return
			end
			Damysos.plotdata(sims, dirname(path); title = title)
		catch e
			@warn "Error for $path"
			showerror(stdout, e)
		end
		
	end
end

Base.global_logger(TerminalLogger())

cmdargs = parse_cmdargs()
files 	= find_files_with_extension(cmdargs["rootpath"],"hdf5")

@progress for f in files
	plotsims(f,cmdargs["maxgraphs"])
end
