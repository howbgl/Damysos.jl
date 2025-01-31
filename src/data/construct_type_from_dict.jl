
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
	for k in keys(BACKWARDSCOMPATLOADABLES)
		m = match(Regex(k),t)
		if !isnothing(m)
			return construct_ngrid_backwards_compat(BACKWARDSCOMPATLOADABLES[k],d)
		end
	end
	throw(ArgumentError(
		"No equivalent for $t found in LOADABLES or BACKWARDSCOMPATLOADABLES."))
end

# Generic method simply extracts primitive numeric values (or Dicts if substructure exists)
# from fieldnames(...)
function construct_type_from_dict(
	t::Type{<:Union{
		SimulationComponent,
		Observable,
		Hamiltonian,
		CartesianKGrid1d,
		CartesianKGrid2d,
		KGrid0d,
		SymmetricTimeGrid}},
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
	ngrid = if "grid" ∈ keys(d)
		construct_type_from_dict(d["grid"])
	elseif "numericalparams" ∈ keys(d)
		construct_type_from_dict(d["numericalparams"])		
	else
		throw(ErrorException("Neither 'grid' nor 'numericalparams' found in dict"))
	end
	return Simulation(
		construct_type_from_dict(d["liouvillian"]),
		construct_type_from_dict(d["drivingfield"]),
		ngrid,
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
