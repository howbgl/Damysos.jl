
export loaddata
export loadlast_testsim
export savedata

const LOADABLES = Dict(
	"Simulation"					=> Simulation,
	"NGrid{.*?}"					=> NGrid,
	"CartesianKGrid2d{.*?}"			=> CartesianKGrid2d,
	"CartesianKGrid2dStrips{.*?}"	=> CartesianKGrid2dStrips,
	"CartesianKGrid1d{.*?}"			=> CartesianKGrid1d,
	"SymmetricTimeGrid{.*?}" 		=> SymmetricTimeGrid,
	"KGrid0d{.*?}"					=> KGrid0d,
	"TwoBandDephasingLiouvillian" 	=> TwoBandDephasingLiouvillian,
	"GappedDirac" 					=> GappedDirac,
	"QuadraticToy"					=> QuadraticToy,
	"BilayerToy"					=> BilayerToy,
	"UnitScaling"					=> UnitScaling,
	"GaussianAPulse" 				=> GaussianAPulse,
	"GaussianEPulse" 				=> GaussianEPulse,
	"GaussianPulse"					=> GaussianPulse,
	"Vector{Observable{.*?}}"		=> Vector{Observable},
	"VelocityX"						=> VelocityX,
	"\bVelocity\b"					=> Velocity,
	"Velocity{.*?}"					=> Velocity,
	"Occupation"					=> Occupation,
	"PowerLawTest"					=> PowerLawTest,
	"LinearTest"					=> LinearTest,
	"ExtendKymaxTest"				=> ExtendKymaxTest
)

const BACKWARDSCOMPATLOADABLES = Dict(
	"NumericalParams2d" 		=> CartesianKGrid2d,
	"NumericalParams1d" 		=> CartesianKGrid1d,
	"NumericalParamsSingleMode" => KGrid0d
)

isloadable(s::String) = [match(Regex(n),s) for n in keys(LOADABLES)] .|> !isnothing |> any
isloadable(object) 	  = isloadable("$(typeof(object))")

function loadable_datatype(s::String)
	for n in keys(LOADABLES) 
		m = match(Regex(n),s)
		if isnothing(m)
			continue
		else
			return LOADABLES[n]
		end
	end
	throw(ArgumentError("No equivalent for $t found in LOADABLES."))
end

function ensurefile_ext(filepath::String,ext::String)
	ext = startswith(ext,".") ? ext : "." * ext
	path, _ext = splitext(filepath)
	_ext == ext && return filepath
	return filepath * ext
end

function savedata(
	sim::Simulation,
	path::String = joinpath(pwd(),getname(sim)))

	@info "Saving simulation data"
	@debug "path = \"$(path)\""

	(success, datapath) = ensuredirpath([path])

	if !success
		@warn "Could not save simulation data to $(datapath)."
		return nothing
	end

	savedata_hdf5(sim, datapath)

	return nothing
end

function savedata(result::ConvergenceTestResult, 
	filepath=joinpath(pwd(),getname(result.test)*"_$(result.test.rtolgoal).hdf5"))

	filepath = ensurefile_ext(filepath,"hdf5")
	h5open(filepath, "cw") do file

		"testresult" ∈ keys(file) && delete_object(file,"testresult")

		g = create_group(file, "testresult")

		g["retcode"]          = Integer(result.retcode)
		g["achieved_atol"]    = result.min_achieved_atol
		g["achieved_rtol"]    = result.min_achieved_rtol
		g["elapsed_time_sec"] = result.elapsed_time_sec
		g["iterations"]       = result.iterations

		savedata_hdf5(result.last_params, g, "last_params")
		savedata_hdf5(result.test, g)

		obs  = result.extrapolated_results
		"extrapolated_results" ∈ keys(file) && delete_object(file,"extrapolated_results")
		gobs = ensuregroup(file, "extrapolated_results")
		gobs["T"] = "$(typeof(obs))"
		for o in obs
			savedata_hdf5(o, gobs)
		end
		close(gobs)
	end
end

function savedata(test::ConvergenceTest, sim::Simulation,
	    filepath=joinpath(pwd(),getname(result.test)*"_$(test.rtolgoal).hdf5"))

	filepath = ensurefile_ext(filepath,"hdf5")
	h5open(filepath, "cw") do file
		savedata_hdf5(sim, ensuregroup(file["completedsims"], sim.id))
	end
end

function construct_ngrid_backwards_compat(
        t::Type{<:Union{CartesianKGrid1d, CartesianKGrid2d, KGrid0d}},
        d::Dict{String})
    timegrid = SymmetricTimeGrid(d["dt"], d["t0"])
    kgrid    = if t == CartesianKGrid1d
        CartesianKGrid1d(d)
    elseif t == CartesianKGrid2d
        CartesianKGrid2d(d)
    elseif t == KGrid0d
        KGrid0d(d)
    else
        throw(ArgumentError("Type $t not supported in construct_ngrid_backwards_compat."))
    end
    return NGrid(kgrid, timegrid)
end

function loadlast_testsim(path::String)
	h5open(path,"r") do file
		g 			= file["completedsims"]
		done_sims 	= [load_obj_hdf5(g[s]) for s in keys(g)]
		
		sort!(done_sims,by=getsimindex)

		isempty(done_sims) && throw(ErrorException(
			"No completed simulation found (test.completedsims is empty)"))
		
		return last(done_sims)
	end
end

function ensuregroup(parent::Union{HDF5.File, HDF5.Group},group::AbstractString)
    return group ∈ keys(parent) ? parent[group] : create_group(parent,group)
end


function replace_data_hdf5(
	parent::Union{HDF5.File,HDF5.Group},
	object::String,
	data)
	if object ∈ keys(parent)
		delete_object(parent[object])
	end
	parent[object] = data
end

function load_obj_hdf5(path::String)
	h5open(path,"r") do file
		return load_obj_hdf5(file)
	end
end

function load_obj_hdf5(object::Union{HDF5.File, HDF5.Group})
	return construct_type_from_dict(read(object))
end

function generic_save_hdf5(object, parent::Union{HDF5.File, HDF5.Group}, grpname::String)
	g = ensuregroup(parent, grpname)
	generic_save_hdf5(object, g)
	close(g)
end

function generic_save_hdf5(object, parent::Union{HDF5.File, HDF5.Group})
	if isloadable(object)
		parent["T"] = "$(typeof(object))"
	end
	for n in fieldnames(typeof(object))
		parent["$n"] = getproperty(object, n)
	end
end
