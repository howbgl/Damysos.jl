
function savedata_hdf5(sim::Simulation,
	path=joinpath(pwd(),getname(sim)))

	filepath = joinpath(path,"data.hdf5")
	filepath = ispath(filepath) ? rename_path(filepath) : filepath
	h5open(filepath, "cw") do file
		savedata_hdf5(sim, file)
	end
	@debug "Saved Simulation data at\n\"$filepath\""
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
	savedata_hdf5(sim.drivingfield,parent)
	close(gdf)
	@debug "Saved driving field"

	
	savedata_hdf5(sim.observables, parent)
	savedata_hdf5(sim.grid, parent)
	savedata_hdf5(sim.liouvillian, parent)
	savedata_hdf5(sim.unitscaling, parent)

	parent["dim"] 		= sim.dimensions
	parent["id"]  		= sim.id
	parent["T"]   		= "Simulation"
end

function savedata_hdf5(df::DrivingField, parent::Union{HDF5.File, HDF5.Group})
	g = ensuregroup(parent, "drivingfield")
	if df isa CompositeDrivingField
		g["T"] = "$(typeof(df))"
		g["prefactors"] = Vector{eltype(df.prefactors)}(df.prefactors)
		gfields = ensuregroup(g, "fields")
		for (i, f) in enumerate(df.fields)
			generic_save_hdf5(f, gfields, "field_$i")
		end
		close(g)
	else
		generic_save_hdf5(df, parent, "drivingfield")
	end
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
	grid::NGrid,
	parent::Union{HDF5.File, HDF5.Group},
	grpname::String = "grid")

	g = create_group(parent, grpname)
	savedata_hdf5(grid.tgrid,g)
	savedata_hdf5(grid.kgrid,g)
	
	g["T"] 		   = "$(typeof(grid))"
	g["tsamples"]  = grid |> gettsamples |> collect
	g["ksamples"]  = grid |> getksamples |> collect
	if grid.kgrid isa CartesianKGrid
		if getdimension(grid) >= 1
			g["kxsamples"] = grid |> getkxsamples |> collect
		end
		if getdimension(grid) == 2
			g["kysamples"] = grid |> getkysamples |> collect
		end
	end
	close(g)
end

function savedata_hdf5(tgrid::SymmetricTimeGrid,parent::Union{HDF5.File,HDF5.Group})
	generic_save_hdf5(tgrid, parent, "tgrid")
end

function savedata_hdf5(kgrid::CartesianKGrid2dStrips, parent::Union{HDF5.File, HDF5.Group})
	return savedata_hdf5(CartesianKGrid2d(kgrid), parent)
end

function savedata_hdf5(kgrid::KGrid,parent::Union{HDF5.File, HDF5.Group})
	generic_save_hdf5(kgrid, parent, "kgrid")
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

function savedata_hdf5(
	dms::DensityMatrixSnapshots{T,U}, 
	parent::Union{HDF5.File, HDF5.Group}) where {T,U<:SMatrix}

	dms_grp = ensuregroup(parent, "density_matrix_snapshots")
	dms_grp["T"] = "$(typeof(dms))"
	dms_grp["density_matrix_type"] = "Vector{SMatrix}"
	dms_grp["tsamples"] = dms.tsamples

	generic_save_hdf5(dms.density_matrices[1].kgrid, dms_grp, "kgrid")

	n,m,l = size(toarray(dms.density_matrices[1]))
	ndms  = length(dms.density_matrices)
	t     = eltype(dms.density_matrices[1].density_matrix[1])
	dspace = dataspace((n,m,l,ndms))

	dm_set = create_dataset(dms_grp, "density_matrices", t, dspace)
	for (i,dm) in enumerate(dms.density_matrices)
		dm_set[:,:,:,i] = toarray(dm)
	end
	close(dm_set)
	close(dms_grp)
end

function savedata_hdf5(t::ConvergenceTest,parent::Union{HDF5.File, HDF5.Group})

	g = ensuregroup(parent, "convergence_parameters")
	if !isempty(t.completedsims)
		params = [currentvalue(t.method, s) for s in t.completedsims]
		g[parametername(t.method)] = params
	end
	close(g)
end

function savedata_hdf5(m::ConvergenceTestMethod,parent::Union{HDF5.File, HDF5.Group})
	return generic_save_hdf5(m,parent,"method")
end

function savedata_hdf5(
	m::Union{PowerLawTest,LinearTest},
	parent::Union{HDF5.File, HDF5.Group},
	grpname::String = "method")
	
	g = ensuregroup(parent, grpname)
	g["T"] 			= "$(typeof(m))"
	g["parameter"] 	= string(m.parameter)

	if m isa PowerLawTest
		g["multiplier"] = m.multiplier
	else # isa LinearTest
		g["shift"] = m.shift
	end

	close(g)
end

function savedata_hdf5(m::ExtendKymaxTest, parent::Union{HDF5.File, HDF5.Group})
	g 		= ensuregroup(parent, "method")
	g["T"] 	= "$(typeof(m))"
	
	savedata_hdf5(m.extendmethod, g, "extendmethod")
	close(g)
end
