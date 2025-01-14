using Damysos, CairoMakie, ArgParse, HDF5, DataFrames, CSV, ProgressLogging, TerminalLoggers

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
            default = "allgoodresults"
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

function extract_pairs(par1::String,par2::String,files::Vector{String})
    return extract_pairs(Regex("$(par1)=([0-9.]+)_$(par2)=([0-9.]+)"),files)
end
function extract_pairs(pattern::Regex,files::Vector{String})
    parameterpairs = Tuple{Float64,Float64}[]
    for f in basename.(files)
        m = match(pattern,f)
        isnothing(m) && throw(ErrorException("Pattern $pattern not matched for $f"))
        mz = (parse(Float64,m.captures[1]),parse(Float64,m.captures[2]))
        push!(parameterpairs,mz)
    end
    return parameterpairs
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


function get_numericalparams(path::String)
	h5open(path,"r") do file
		if "testresults" ∈ keys(file)
			return Damysos.load_obj_hdf5(file["last_params"])
		elseif any(isnothing.(match.(r"start_*",keys(file))) .|> !)
			for k in keys(file)
				!isnothing(match(r"start_*",k)) && return Damysos.load_obj_hdf5(
					file[k],"numericalparams")
			end
		else
			return nothing
		end
	end
end


function collectresults(args::Dict)
	p1 = args["parameter1"]
	p2 = args["parameter2"]
	r1 = args["regex1"] == "automatic" ? Regex(p1*"=([0-9.]+)") : Regex(args["regex1"])
	r2 = args["regex2"] == "automatic" ? Regex(p2*"=([0-9.]+)") : Regex(args["regex1"])
	return collectresults(load_hdf5_files(args["rootpath"]),p1,r1,p2,r2)
end


function collectresults(files::Vector{String},p1::String,r1::Regex,p2::String,r2::Regex)

	goodfiles = String[]
	coord = Tuple{Float64,Float64}[]
	p1s = Float64[]
	p2s = Float64[]
	dts = Float64[]
	dkxs = Float64[]
	rtols = Float64[]
	atols = Float64[]

	@progress for f in files
		
		if isnothing(match(r1,f)) || isnothing(match(r2,f)) 
			@warn "No regex match for $r1 & $r2 in $f"
			continue
		end
		
		_p1 = parse(Float64, match(r1,f).captures[1])
		_p2 = parse(Float64, match(r2,f).captures[1])

		if (_p1,_p2) ∈ coord
			@warn "Duplicate at at  $p1 = $_p1 $p2 = $(_p2).Skipping..."
			continue
		end

		h5open(f,"r") do file
			if "testresult" ∈ keys(file)
				if successful_retcode(ReturnCode.T(read(file["testresult"],"retcode")))
					params = Damysos.load_obj_hdf5(file["testresult/last_params"])

					push!(rtols,read(file["testresult"],"achieved_rtol"))
					push!(atols,read(file["testresult"],"achieved_atol"))
					push!(dkxs,params.dkx)
					push!(dts,params.dt)
					push!(goodfiles,f)
					push!(p2s,_p2)
					push!(p1s,_p1)
					push!(coord,(_p1,_p2))
				else
					@info "Result not converged at at $p1 = $_p1 $p2 = $_p2 in $f"
				end

			else
				@warn "testresult not found at  $p1 = $_p1 $p2 = $_p2 in $f"
			end
		end

	end

	return DataFrame(
		:path => goodfiles,
		Symbol(p1) => p1s,
		Symbol(p2) => p2s,
		:dt => dts,
		:dkx => dkxs,
		:rtol => rtols,
		:atol => atols)
end

cmdargs = parse_cmdargs()
data 	= collectresults(cmdargs)

CSV.write(cmdargs["outputfile"]*".csv",data)

function loadlastobs(path::String,convergencparameter::String)

	file = h5open(path,"r")
	pat  = Regex(convergencparameter*"=([0-9.]+)")
	sorting_trafo(x) = isnothing(x) ? Inf : parse(Float64,x.captures[1])

	ms = [match(pat,f) for f in keys(file)]
	msorted = sort(ms,by=sorting_trafo)
	lastsim = file[msorted[1].match]
	obsdict = read(lastsim,"observables")
	obs 	= Damysos.Observable[]
	for (k,v) in obsdict
		if k == "velocity"
			push!(obs,Velocity(v))
		elseif k == "occupation"
			push!(obs,Occupation(v))
		end
	end
	close(file)
	return obs
end

function loadlastsim(path::String,convergencparameter::String)
	file = h5open(path,"r")
	pat  = Regex(convergencparameter*"=([0-9.]+)")
	sorting_trafo(x) = isnothing(x) ? Inf : parse(Float64,x.captures[1])

	ms = [match(pat,f) for f in keys(file)]
	msorted = sort(ms,by=sorting_trafo)
	lastsim = read(file,msorted[1].match)
	close(file)
	return lastsim
end

function savegoodsims(data::DataFrame,outpath::String,convergencparameter::String)
	file = h5open(outpath*".hdf5","cw")

	@progress for r in eachrow(data)
		source 				= h5open(r.path,"r")
		pat  				= Regex(convergencparameter*"=([0-9.]+)")
		sorting_trafo(x) 	= isnothing(x) ? Inf : parse(Float64,x.captures[1])

		ms 			= [match(pat,f) for f in keys(source)]
		msorted 	= sort(ms,by=sorting_trafo)
		groupname 	= "zeta=$(r.zeta)_M=$(r.M)"

		copy_object(source,msorted[1].match,file,groupname)
		file[groupname]["zeta"] = r.zeta
		file[groupname]["M"] 	= r.M

		close(source)
	end
	close(file)
end
