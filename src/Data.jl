
export loaddata
export loadlast_testsim
export savedata

const LOADABLES = Dict(
	"Simulation"					=> Simulation,
	"NumericalParams2d" 			=> NumericalParams2d,
	"NumericalParams1d" 			=> NumericalParams1d,
	"NumericalParamsSingleMode" 	=> NumericalParamsSingleMode,
	"TwoBandDephasingLiouvillian" 	=> TwoBandDephasingLiouvillian,
	"GappedDirac" 					=> GappedDirac,
	"QuadraticToy"					=> QuadraticToy,
	"UnitScaling"					=> UnitScaling,
	"GaussianAPulse" 				=> GaussianAPulse,
	"GaussianEPulse" 				=> GaussianEPulse,
	"GaussianPulse"					=> GaussianPulse,
	"Vector{Observable{.*?}}"		=> Vector{Observable},
	"Velocity" 						=> Velocity,
	"VelocityX"						=> VelocityX,
	"Occupation"					=> Occupation,
	"PowerLawTest"					=> PowerLawTest,
	"LinearTest"					=> LinearTest
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
	ext = replace(ext,"." => "")
	path = splitext(filepath)[1]
	return path * "." * ext
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

		generic_save_hdf5(result.last_params, g, "last_params")
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

function savedata_hdf5(sim::Simulation,
	path=joinpath(pwd(),getname(sim)))

	filepath = joinpath(path,"data.hdf5")
	filepath = ispath(filepath) ? rename_path(filepath) : filepath
	h5open(filepath, "cw") do file
		savedata_hdf5(sim, file)
	end
	@debug "Saved Simulation data at\n\"$datapath\""
end

function savedata_hdf5(
	sim::Simulation,
	parent::Union{HDF5.File, HDF5.Group})


	df  = sim.drivingfield
	ts  = gettsamples(sim)
	gdf = create_group(parent, "drivingfield")
	@debug "Created group $gdf"
	for (f, n) in zip(
		(t -> efieldx(df, t), t -> efieldy(df, t), t -> vecpotx(df, t), t -> vecpoty(df, t)),
		("fx", "fy", "ax", "ay"))
		
		gdf[n] = f.(ts)
	end
	generic_save_hdf5(sim.drivingfield,gdf)
	close(gdf)
	@debug "Saved driving field"

	
	savedata_hdf5(sim.observables, parent)
	savedata_hdf5(sim.numericalparams, parent)
	savedata_hdf5(sim.liouvillian, parent)
	savedata_hdf5(sim.unitscaling, parent)

	parent["dim"] 		= sim.dimensions
	parent["id"]  		= sim.id
	parent["T"]   		= "Simulation"
end

function savedata_hdf5(
	obs::Vector{<:Observable},
	parent::Union{HDF5.File, HDF5.Group})
	
	gobs = create_group(parent, "observables")
	gobs["T"] = "$(typeof(obs))"
	for o in obs
		savedata_hdf5(o, gobs)
	end
	close(gobs)
end

function savedata_hdf5(
	p::NumericalParameters,
	parent::Union{HDF5.File, HDF5.Group})

	g = create_group(parent, "numericalparams")
	generic_save_hdf5(p, g)
	g["tsamples"]  = p |> gettsamples |> collect
	g["kxsamples"] = p |> getkxsamples |> collect
	if p isa NumericalParams2d
		g["kysamples"] = p |> getkysamples |> collect
	end
	close(g)
end

function savedata_hdf5(us::UnitScaling, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(us, parent, "unitscaling")
end

function savedata_hdf5(l::TwoBandDephasingLiouvillian, parent::Union{HDF5.File, HDF5.Group})
	g = create_group(parent, "liouvillian")
	g["T"]  = "TwoBandDephasingLiouvillian"
	g["t1"] = l.t1
	g["t2"] = l.t2
	savedata_hdf5(l.hamiltonian, g)
	close(g)
end

function savedata_hdf5(h::GeneralTwoBand, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(h, parent, "hamiltonian")
end

function savedata_hdf5(v::Velocity, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(v, parent, "velocity")
end

function savedata_hdf5(v::VelocityX, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(v, parent, "velocity_x")
end

function savedata_hdf5(o::Occupation, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(o, parent, "occupation")
end

function savedata_hdf5(t::ConvergenceTest,parent::Union{HDF5.File, HDF5.Group})

	g = ensuregroup(parent, "convergence_parameters")
	if !isempty(t.completedsims)
		params = [currentvalue(t.method, s) for s in t.completedsims]
		g[string(t.method.parameter)] = params
	end
	close(g)
end

function savedata_hdf5(m::ConvergenceTestMethod,parent::Union{HDF5.File, HDF5.Group})
	return generic_save_hdf5(m,parent,"method")
end

function savedata_hdf5(
	m::Union{PowerLawTest,LinearTest},
	parent::Union{HDF5.File, HDF5.Group})
	
	g = ensuregroup(parent,"method")
	g["T"] 			= "$(typeof(m))"
	g["parameter"] 	= string(m.parameter)

	if m isa PowerLawTest
		g["multiplier"] = m.multiplier
	else # isa LinearTest
		g["shift"] = m.shift
	end

	close(g)
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

function construct_type_from_dict(d::Dict{String})
	return construct_type_from_dict(d["T"],d)
end

function construct_type_from_dict(t::String,d::Dict{String})
	for k in keys(LOADABLES)
		m = match(Regex(k),t)
		if !isnothing(m)
			return construct_type_from_dict(LOADABLES[k],d)
		end
	end
	throw(ArgumentError("No equivalent for $t found in LOADABLES."))
end

# Generic method simply extracts primitive numeric values (or Dicts if substructure exists)
# from fieldnames(...)
function construct_type_from_dict(
	t::Type{<:Union{SimulationComponent,Observable,Hamiltonian}},
	d::Dict{String})

    names = String.(fieldnames(t))
    args = []
    for n in names
        if n ∈ keys(d)
            field = d[n]
            if field isa Dict
                push!(args,construct_type_from_dict(field["T"],field))
            else
                push!(args,d[n])
            end
        else
            throw(KeyError(n))
        end
    end
    return t(args...)
end

function construct_type_from_dict(::Type{<:Simulation},d::Dict{String})
	return Simulation(
		construct_type_from_dict(d["liouvillian"]),
		construct_type_from_dict(d["drivingfield"]),
		construct_type_from_dict(d["numericalparams"]),
		[construct_type_from_dict(d["observables"])...], # Vector{Obs} => Vector{Obs{T}}
		construct_type_from_dict(d["unitscaling"]),
		d["id"],
		d["dim"])
end

function construct_type_from_dict(::Type{Vector{Observable}},d::Dict{String})
	obs = Observable[]
	for o in values(d)
		# avoid trying to load obs["T"] = "Vector{Observable{...}}"
		o isa Dict && push!(obs,construct_type_from_dict(o["T"],o))
	end
	return obs
end

function construct_type_from_dict(::Type{<:PowerLawTest},d::Dict{String})
	return PowerLawTest(Symbol(d["parameter"]),d["multiplier"])
end

function construct_type_from_dict(::Type{<:LinearTest},d::Dict{String})
	return LinearTest(Symbol(d["parameter"]),d["shift"])
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
