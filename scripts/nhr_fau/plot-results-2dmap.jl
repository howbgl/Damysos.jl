using Damysos, CairoMakie, ArgParse, HDF5, DataFrames, CSV, TerminalLoggers, ProgressLogging

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
		"--dimension", "-d"
			help = "Dimension of the simulations"
			arg_type = Int
			default = 2
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
            default = "resultmap"
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

	@progress for (i, f) in enumerate(files)
		h5open(f, "r") do file
			rcodes[i] = "testresult" ∈ keys(file) ? read(file["testresult"], "retcode") : 4
		end
	end
	return rcodes
end

function get_retcode(filepath::String)
	h5open(filepath, "r") do file
		return "testresult" ∈ keys(file) ? read(file["testresult"], "retcode") : 4
	end
end

function get_min_achieved_rtol(files::Vector{String})
	
	rtols = zeros(Float64, length(files))

	for (i,f) in enumerate(files)
		h5open(f,"r") do file
			rtols[i] = "testresult" ∈ keys(file) ? read(file["testresult"], "achieved_rtol") : Inf
		end
	end
	return rtols
end

function get_numericalparams(path::String,dimension::Integer)

	getpars = if dimension==2
		x -> NumericalParams2d(read(x,"last_params"))
	elseif dimension==1
		x -> NumericalParams1d(read(x,"last_params"))
	elseif dimension==0
		x -> NumericalParamsSingleMode(read(x,"last_params"))
	else
		throw(ErrorException(
			"Wrong dimension $dimension given. Can only be 0,1 or 2."))
	end 

	h5open(path,"r") do file
		if "testresult" ∈ keys(file)
			return getpars(file["testresult"])
		else
			throw(ErrorException("no group testresult found in file $path"))
		end
	end
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
		args["outputfile"] * "_retcode.png",
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

	@show retcodes

	@info "plotting"

	fig = Figure()
	ax, hm = heatmap(
		fig[1, 1],
		pars1,
		pars2,
		retcodes,
		colorrange = (0, 4),
		axis = (; xscale = log10, yscale = log10, xlabel = name1, ylabel = name2)
		)

	Colorbar(fig[1, 2], hm, ticks = (0:4, string.(ReturnCode.T.(0:4))))
	CairoMakie.save(outpath, fig)
	return nothing
end

function plot_min_achieved_rtol(args::Dict{String, Any})
	p1 = args["parameter1"]
	p2 = args["parameter2"]
	pattern1 = args["regex1"]
	pattern2 = args["regex2"]
	pattern1 = pattern1 == "automatic" ? "_$(p1)=([0-9.]+)_" : pattern1
	pattern2 = pattern2 == "automatic" ? "_$(p2)=([0-9.]+)_" : pattern2

	return plot_min_achieved_rtol(
		args["rootpath"],
		args["outputfile"] * "_rtols.png",
		pattern1,
		pattern2,
		p1,
		p2)
end

function plot_min_achieved_rtol(
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

	@info "getting achieved rtols"
	rtols = get_min_achieved_rtol(files)
	rtols = [isfinite(x) ? x : NaN for x in rtols]

    length(rtols) == 0 && return nothing

	@info "plotting"

	fig = Figure()
	ax, hm = heatmap(
		fig[1, 1],
		pars1,
		pars2,
		rtols,
		axis = (; xscale = log10, yscale = log10, xlabel = name1, ylabel = name2))

	Colorbar(fig[1, 2], hm)
	CairoMakie.save(outpath, fig)
	return nothing
end

function save_all_results(args::Dict{String,Any})
	p1 = args["parameter1"]
	p2 = args["parameter2"]
	pattern1 = args["regex1"]
	pattern2 = args["regex2"]
	pattern1 = pattern1 == "automatic" ? "_$(p1)=([0-9.]+)_" : pattern1
	pattern2 = pattern2 == "automatic" ? "_$(p2)=([0-9.]+)_" : pattern2
	return save_all_results(
		args["rootpath"],
		args["outputfile"] * ".csv",
		pattern1,
		pattern2,
		p1,
		p2,
		args["dimension"])
end

function save_all_results(
	rootpath::String,
	outpath::String,
	pattern1::String,
	pattern2::String,
	name1::String,
	name2::String,
	dimension::Integer)

	files = load_hdf5_files(rootpath)

	codes = get_retcodes(files)

	@info "extracting parameters"
	pars1 = extract_parameter(pattern1, files)
	pars2 = extract_parameter(pattern2, files)

	
	pars = [get_numericalparams(f,dimension) for f in files]
	dts = [isnothing(p) ? missing : p.dt for p in pars]
	dkxs = [isnothing(p) ? missing :  p.dkx for p in pars]
	kxmaxs = [isnothing(p) ? missing :  p.kxmax for p in pars]

	df = DataFrame(
		name1 => pars1, 
		name2 => pars2, 
		"dt" => dts, 
		"dkx" => dkxs, 
		"returncode" => codes,
		"kxmax" => kxmaxs)

	if dimension == 2
		dkys  = [isnothing(p) ? missing :  p.dky for p in pars]
		kymax = [isnothing(p) ? missing :  p.kymax for p in pars]
		df[!,"dky"] = dkys
		df[!,"kymax"] = kymax
	end
	
	CSV.write(outpath,df)

	return nothing
end

function save_failed_results(args::Dict{String,Any})
	p1 = args["parameter1"]
	p2 = args["parameter2"]
	pattern1 = args["regex1"]
	pattern2 = args["regex2"]
	pattern1 = pattern1 == "automatic" ? "_$(p1)=([0-9.]+)_" : pattern1
	pattern2 = pattern2 == "automatic" ? "_$(p2)=([0-9.]+)_" : pattern2
	return save_failed_results(
		args["rootpath"],
		args["outputfile"] * "_failed.csv",
		pattern1,
		pattern2,
		p1,
		p2,
		args["dimension"])
end

function save_failed_results(
	rootpath::String,
	outpath::String,
	pattern1::String,
	pattern2::String,
	name1::String,
	name2::String,
	dimension::Integer)
	
	files = load_hdf5_files(rootpath)

	codes = get_retcodes(files)
	i = (!).(successful_retcode.(ReturnCode.T.(codes)))

	@info "extracting parameters"
	pars1 = extract_parameter(pattern1, files[i])
	pars2 = extract_parameter(pattern2, files[i])

	
	pars = [get_numericalparams(f,dimension) for f in files[i]]
	dts = [isnothing(p) ? missing : p.dt for p in pars]
	dkxs = [isnothing(p) ? missing :  p.dkx for p in pars]
	kxmaxs = [isnothing(p) ? missing :  p.kxmax for p in pars]

	df = DataFrame(
		name1 => pars1, 
		name2 => pars2, 
		"dt" => dts, 
		"dkx" => dkxs, 
		"returncode" => codes[i],
		"kxmax" => kxmaxs)

	if dimension == 2
		dkys  = [isnothing(p) ? missing :  p.dky for p in pars]
		kymax = [isnothing(p) ? missing :  p.kymax for p in pars]
		df[!,"dky"] = dkys
		df[!,"kymax"] = kymax
	end
	
	CSV.write(outpath,df)

	return nothing
end

cmdargs = parse_cmdargs()

save_failed_results(cmdargs)
save_all_results(cmdargs)
plot_retcodes(cmdargs)
plot_min_achieved_rtol(cmdargs)