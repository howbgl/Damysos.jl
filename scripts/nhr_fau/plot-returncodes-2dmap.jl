using Damysos, CairoMakie, ArgParse, HDF5

function parse_cmdargs()
	s = ArgParseSettings()
	@add_arg_table! s begin
		"rootpath"
            help = "Root path containing all .hdf5 files to be loaded"
            arg_type = String
            required = true
		"parameter1"
            help = "Name of parameter 1"
            arg_type = String
            required = true
		"parameter2"
            help = "Name of parameter 2"
            arg_type = String
            required = true
		"--regex1"
            help = "Regex pattern for fetching parameter 1"
            arg_type = String
            default = "automatic"
		"--regex2"
            help = "Regex pattern for fetching parameter 2"
            arg_type = String
            default = "automatic"
		"--outputfile", "-o"
            help = "Name of the output file"
            arg_type = String
            default = "returncode_map.png"
	end
	return parse_args(s)
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

function extract_parameter(pattern::Union{String, Regex}, rootpath::String)
	return extract_parameter(pattern, load_hdf5_files(rootpath))
end
function extract_parameter(pattern::String, files::Vector{String})
	return extract_parameter(Regex(pattern), files)
end
function extract_parameter(pattern::Regex, files::Vector{String})

	parameters = Float64[]
	for file in files
		m = match(pattern, file)
		if isnothing(m)
			@warn "No regex match for $(pattern) in $file"
			continue
		end
		push!(parameters, parse(Float64, m.captures[1]))

	end
	return parameters
end

function get_retcodes(files::Vector{String})

	rcodes = zeros(Int, length(files))

	for (i, f) in enumerate(files)
		h5open(f, "r") do file
			rcodes[i] = "testresult" ∈ keys(file) ? read(file["testresult"], "retcode") : 4
		end
	end
	return rcodes
end

function plot_retcodes(args::Dict{String, Any})

	p1 = args["parameter1"]
	p2 = args["parameter2"]
	pattern1 = args["regex1"]
	pattern2 = args["regex2"]
	pattern1 = pattern1 == "automatic" ? "_$(p1)=([0-9.]+)_" : pattern1
	pattern2 = pattern2 == "automatic" ? "_$(p2)=([0-9.]+)_" : pattern2

	return plot_retcodes(
		args["rootpath"],
		args["outputfile"],
		pattern1,
		pattern2,
		p1,
		p2)
end

function plot_retcodes(
	rootpath::String,
	outpath::String,
	pattern1::String,
	pattern2::String,
	name1::String,
	name2::String)

	files = load_hdf5_files(rootpath)

	@info "extracting parameters"
	pars1 = extract_parameter(pattern1, files)
	pars2 = extract_parameter(pattern2, files)

	@info "getting returncodes"
	retcodes = get_retcodes(files)

    length(retcodes) == 0 && return nothing

	@info "plotting"

	fig = Figure()
	ax, hm = heatmap(
		fig[1, 1],
		pars1,
		pars2,
		retcodes,
		colorrange = (0, 4),
		axis = (; xscale = log10, yscale = log10, xlabel = name1, ylabel = name2))

	Colorbar(fig[1, 2], hm, ticks = (0:4, string.(ReturnCode.T.(0:4))))
	CairoMakie.save(outpath, fig)
	return nothing
end

plot_retcodes(parse_cmdargs())
