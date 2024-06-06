
export load
export loaddata
export save
export savedata
export savemetadata

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

		g["retcode"]            = Integer(result.retcode)
		g["achieved_atol"]      = result.min_achieved_atol
		g["achieved_rtol"]      = result.min_achieved_rtol
		g["elapsed_time_sec"]   = result.elapsed_time_sec
		g["iterations"]         = result.iterations

		generic_save_hdf5(result.last_params,g,"last_params")
		savedata_hdf5(result.test.method, result.test, g)
	end
end

function savedata(test::ConvergenceTest, sim::Simulation)

	h5open(test.testdatafile, "cw") do file
		savedata_hdf5(sim, create_group(file, sim.id))
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
	close(gdf)
	@debug "Saved driving field"

	gobs = create_group(parent, "observables")
	for o in sim.observables
		savedata_hdf5(o, gobs)
	end
	close(gobs)

	savedata_hdf5(sim.numericalparams, parent)
	savedata_hdf5(sim.liouvillian, parent)
	savedata_hdf5(sim.unitscaling, parent)
	parent["dim"] = sim.dimensions
	parent["id"]  = sim.id
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

function savedata_hdf5(
	m::Union{PowerLawTest, LinearTest},
	t::ConvergenceTest,
	parent::Union{HDF5.File, HDF5.Group})

	g = create_group(parent, "convergence_parameters")
	params = [currentvalue(m, s) for s in t.completedsims]
	g[string(m.parameter)] = params
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

function loadlastparams(filepath::String,::Type{T}) where {T<:NumericalParameters}
	h5open(filepath,"r") do file
		return T(read(file["testresult"],"last_params"))
	end
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
	for n in fieldnames(typeof(object))
		parent["$n"] = getproperty(object, n)
	end
end

function add_drivingfield!(dat::DataFrame, df::DrivingField, tsamples::AbstractVector{<:Real})

	fx = get_efieldx(df)
	fy = get_efieldy(df)
	ax = get_vecpotx(df)
	ay = get_vecpoty(df)

	dat.fx = fx.(tsamples)
	dat.fy = fy.(tsamples)
	dat.ax = ax.(tsamples)
	dat.ay = ay.(tsamples)
end

function savemetadata(sim::Simulation)

	filename            = "simulation.meta"
	altpath             = joinpath(pwd(), basename(sim.datapath))
	(success, datapath) = ensurepath([sim.datapath, altpath])
	if success
		if save(joinpath(datapath, filename), sim)
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

