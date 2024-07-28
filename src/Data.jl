
export load
export loaddata
export save
export savedata
export savemetadata

const LOADABLES = Dict(
	"Simulation"					=> Simulation,
	"NumericalParams2d" 			=> NumericalParams2d,
	"NumericalParams1d" 			=> NumericalParams1d,
	"NumericalParamsSingleMode" 	=> NumericalParamsSingleMode,
	"TwoBandDephasingLiouvillian" 	=> TwoBandDephasingLiouvillian,
	"GappedDirac" 					=> GappedDirac,
	"UnitScaling"					=> UnitScaling,
	"GaussianAPulse" 				=> GaussianAPulse,
	"GaussianEPulse" 				=> GaussianEPulse,
	"GaussianPulse"					=> GaussianPulse,
	"Vector{Observable{.*?}}"		=> Vector{Observable},
	"Velocity" 						=> Velocity,
	"Occupation"					=> Occupation
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

function savedata(
	sim::Simulation;
	altpath = joinpath(pwd(), basename(sim.datapath)),
	savecsv = true,
	savehdf5 = true)

	@info "Saving simulation data"
	@debug "datapath = \"$(sim.datapath)\""

	datapath            = sim.datapath
	(success, datapath) = ensurepath([datapath, altpath])

	if !success
		@warn "Could not save simulation data to $(sim.datapath) or $altpath."
		return nothing
	end

	savemetadata(sim)
	savecsv && savedata_csv(sim, datapath)
	savehdf5 && savedata_hdf5(sim, datapath)

	return nothing
end

function savedata(result::ConvergenceTestResult)

	h5open(result.test.testdatafile, "cw") do file
		g = create_group(file, "testresult")

		g["retcode"]          = Integer(result.retcode)
		g["achieved_atol"]    = result.min_achieved_atol
		g["achieved_rtol"]    = result.min_achieved_rtol
		g["elapsed_time_sec"] = result.elapsed_time_sec
		g["iterations"]       = result.iterations

		start = create_group(file,"start")
		savedata_hdf5(result.test.start,start)

		generic_save_hdf5(result.last_params, g, "last_params")
		savedata_hdf5(result.test, g)
	end
end

function savedata(test::ConvergenceTest, sim::Simulation)

	h5open(test.testdatafile, "cw") do file
		savedata_hdf5(sim, create_group(file["completedsims"], sim.id))
	end
end

function savedata_hdf5(sim::Simulation, datapath::String)

	rename_file_if_exists(joinpath(datapath, "data.hdf5"))
	h5open(joinpath(datapath, "data.hdf5"), "cw") do file
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

	gobs 		= create_group(parent, "observables")
	for o in sim.observables
		savedata_hdf5(o, gobs)
	end
	close(gobs)

	savedata_hdf5(sim.numericalparams, parent)
	savedata_hdf5(sim.liouvillian, parent)
	savedata_hdf5(sim.unitscaling, parent)

	parent["dim"] 		= sim.dimensions
	parent["id"]  		= sim.id
	parent["datapath"] 	= sim.datapath
	parent["plotpath"] 	= sim.plotpath
	parent["T"]   		= "Simulation"
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

function savedata_hdf5(o::Occupation, parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(o, parent, "occupation")
end

function savedata_hdf5(t::ConvergenceTest,parent::Union{HDF5.File, HDF5.Group})

	g = create_group(parent, "convergence_parameters")
	if !isempty(t.completedsims)
		params = [currentvalue(t.method, s) for s in t.completedsims]
		g[string(t.method.parameter)] = params
	end
	close(g)
end

function savedata_csv(sim::Simulation, datapath::String)

	tsamples = getparams(sim).tsamples
	dat      = DataFrame(t = tsamples)

	for o in sim.observables
		add_observable!(dat, o)
		add_drivingfield!(dat, sim.drivingfield, tsamples)
	end
	CSV.write(joinpath(datapath, "data.csv"), dat)
	@debug "Saved data (CSV) at $(joinpath(datapath,"data.csv"))"
end

function loaddata(sim::Simulation)
	return DataFrame(CSV.File(joinpath(sim.datapath, "data.csv")))
end

function loadsimulation_hdf5(path::String)
	h5open(path,"r") do file
		loadsimulation_hdf5(file)
	end
end

function loadsimulation_hdf5(parent::Union{HDF5.File, HDF5.Group})

	ldict 	= read(parent,"liouvillian")
	dfdict 	= read(parent,"drivingfield")
	pdict 	= read(parent,"numericalparams")
	usdict 	= read(parent,"unitscaling")
	odict  	= read(parent,"observables")

	l  = construct_type_from_dict(ldict["T"],ldict)
	df = construct_type_from_dict(dfdict["T"],dfdict)
	p  = construct_type_from_dict(pdict["T"],pdict)
	us = construct_type_from_dict(usdict["T"],usdict)

	obs = Vector{Observable}(l)
	
	for o in values(odict)
		push!(obs,construct_type_from_dict(o["T"],o))
	end
	return Simulation(
		l,
		df,
		p,
		obs,
		us,
		read(parent,"id"),
		read(parent,"datapath"),
		read(parent,"plotpath"),
		read(parent,"dim"))
end

function loadlastparams(filepath::String, ::Type{T}) where {T <: NumericalParameters}
	h5open(filepath, "r") do file
		return T(read(file["testresult"], "last_params"))
	end
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

function construct_type_from_dict(t::Union{DataType,UnionAll},d::Dict{String})
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

function add_observable!(dat::DataFrame, v::Velocity)
	dat.vx      = v.vx
	dat.vxintra = v.vxintra
	dat.vxinter = v.vxinter
	# skip for 1d
	if length(v.vy) == length(v.vx)
		dat.vy      = v.vy
		dat.vyintra = v.vyintra
		dat.vyinter = v.vyinter
	end

end

function add_observable!(dat::DataFrame, occ::Occupation)
	dat.cbocc = occ.cbocc
end

function generic_save_hdf5(object, parent::Union{HDF5.File, HDF5.Group}, grpname::String)
	g = create_group(parent, grpname)
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

function add_drivingfield!(
	dat::DataFrame,
	df::DrivingField, 
	tsamples::AbstractVector{<:Real})

	fx = get_efieldx(df)
	fy = get_efieldy(df)
	ax = get_vecpotx(df)
	ay = get_vecpoty(df)

	dat.fx = fx.(tsamples)
	dat.fy = fy.(tsamples)
	dat.ax = ax.(tsamples)
	dat.ay = ay.(tsamples)
end

function savemetadata(sim::Simulation;
	save_observables = false,
	altpath = joinpath(pwd(), basename(sim.datapath)),
	filename = "simulation.meta")


	(success, datapath) = ensurepath([sim.datapath, altpath])
	_sim = save_observables ? sim : Simulation(
		sim.liouvillian,
		sim.drivingfield,
		sim.numericalparams,
		empty(sim.observables),
		sim.unitscaling,
		sim.id,
		sim.datapath,
		sim.plotpath,
		sim.dimensions)
	if success
		if save(joinpath(datapath, filename), _sim)
			@debug "Simulation metadata saved at \"" * joinpath(datapath, filename) * "\""
			return
		end
	end

	@warn "Could not save simulation metadata."
end


function savemetadata(ens::Ensemble)

	filename            = "ensemble.meta"
	altpath             = joinpath(pwd(), basename(ens.datapath))
	(success, datapath) = ensurepath([ens.datapath, altpath])
	if success
		if save(joinpath(datapath, filename), ens)
			@debug "Ensemble metadata saved at \"" * joinpath(datapath, filename) * "\""
			return
		end
	end

	@warn "Could not save ensemble metadata."
end


function save(filepath::String, object)

	try
		touch(filepath)
		file = open(filepath, "w")
		write(file, "$object")
		close(file)
	catch e
		@warn "Could not save to $filepath ", e
		return false
	end
	return true
end


function load(filepath::String)

	file = open(filepath, "r")
	code = read(file, String)
	close(file)

	return eval(Meta.parse(code))
end

