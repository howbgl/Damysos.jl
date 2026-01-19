export Simulation


"""
	Simulation{T}(l, df, g, obs, us, d[, id])

Represents a simulation with all physical and numerical parameters specified.

# Fields
- `l::Liouvillian{T}`: describes physical system via Liouville operator
- `df::DrivingField{T}`: laser field driving the system
- `g::NGrid{T}`: time & reciprocal (k-) space discretization
- `obs::Vector{Observable{T}}`: physical observables to be computed
- `us::UnitScaling{T}`: time- and lengthscale linking dimensionless units to SI units
- `id::String`: identifier of the Simulation
- `dimensions::UInt8`: system can be 0d (single mode),1d or 2d

# See also
[`NGrid`](@ref), [`TwoBandDephasingLiouvillian`](@ref), [`UnitScaling`](@ref),
[`Velocity`](@ref), [`Occupation`](@ref), [`GaussianAPulse`](@ref)
"""
struct Simulation{T <: Real}
	liouvillian::Liouvillian{T}
	drivingfield::DrivingField{T}
	grid::NGrid{T}
	observables::Vector{Observable{T}}
	unitscaling::UnitScaling{T}
	id::String
	dimensions::UInt8
	function Simulation{T}(l, df, g, obs, us, id, d) where {T <: Real}
		check_compatibility(l, df, g, obs, us, id, d)
		new(l, df, g, obs, us, id, getdimension(g))
	end
end

function check_compatibility(
	l::Liouvillian, 
	df::DrivingField, 
	g::NGrid, 
	obs::Vector{<:Observable}, 
	us::UnitScaling, 
	id::String, 
	d::Integer)
	if d != getdimension(g)
		@warn """
		The dimension d=$d does not match the the numerical k grid.
		Overwriting to d=$(getdimension(g)) instead."""
	end
	if isperiodic(l) && !(g.kgrid isa PeriodicKGrid)
		throw(ArgumentError("Liouvillian is periodic, but k-grid is not periodic."))
	end
	return nothing
end

Simulation(path::String) = load_obj_hdf5(path)

function Simulation(
	l::Liouvillian{T},
	df::DrivingField{T},
	g::NGrid{T},
	obs::Vector{<:Observable{T}},
	us::UnitScaling{T},
	id::String,
	d::Integer = getdimension(g)) where {T <: Real}

	return Simulation{T}(l, df, g, [obs...], us, id, UInt8(d))
end

function Simulation(
	l::Liouvillian,
	df::DrivingField,
	g::NGrid,
	obs::Vector{<:Observable},
	us::UnitScaling,
	id = string(hash([l, df, g, obs, us]), base = 16))
	
	return Simulation(l, df, g, [obs...], us, id)
end


function Base.show(io::IO, ::MIME"text/plain", s::Simulation{T}) where T

	buf = IOBuffer()
	print(io, "Simulation{$T} ($(s.dimensions)d):\n")

	for n in fieldnames(Simulation{T})
		if !(n == :dimensions)
			if n == :observables
				println(io, " Observables:")
				str = join([getshortname(o) for o in getfield(s, n)], "\n")
				println(io, prepend_spaces(str, 2))
			elseif getfield(s, n) isa SimulationComponent
				Base.show(buf, MIME"text/plain"(), getfield(s, n))
				str = String(take!(buf))
				print(io, prepend_spaces(str) * "\n")
			else
				Base.show(buf, MIME"text/plain"(), getfield(s, n))
				str = String(take!(buf))
				println(io, " $n: " * str)
			end
		end
	end
end

function Base.show(io::IO, ::MIME"text/plain", c::Union{SimulationComponent, Hamiltonian})
	println(io, getshortname(c) * ":")
	print(io, printfields_generic(c) |> prepend_spaces)
end


for func ∈ (BAND_SYMBOLS..., DIPOLE_SYMBOLS..., VELOCITY_SYMBOLS...)
	@eval(Damysos, $func(s::Simulation) = $func(s.liouvillian))
end

function Base.isapprox(
	s1::Simulation{T},
	s2::Simulation{U};
	atol::Real = 0,
	rtol = atol > 0 ? 0 : √eps(promote_type(T, U)),
	nans::Bool = false) where {T, U}

	allobs = []
	for obs1 in s1.observables
		for obs2 in s2.observables 
			if typeof(obs1) == typeof(obs2)
				push!(allobs, (obs1, obs2))
				break
			end
		end
	end
	if length(allobs) != length(s1.observables) || length(allobs) != length(s2.observables)
		return false
	end

	return all([Base.isapprox(o1, o2; 
        atol = atol, rtol = rtol, nans = nans) for (o1, o2) in allobs])
end

function add_observable!(sim::Simulation, ::Type{O}) where {O <: Observable}
	sim.observables = [sim.observables... , O(sim)]
	return nothing	
end
function add_observable!(sim::Simulation, o::Observable)
	push!(sim.observables, o)
	return nothing	
end

function getshortname(sim::Simulation{T}) where {T <: Real}
	return "Simulation{$T}($(sim.dimensions)d)" * getshortname(sim.liouvillian) * "_" *
		   getshortname(sim.drivingfield)
end


getnames_obs(sim::Simulation) = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation) = vcat(arekresolved.(sim.observables)...)
getname(sim::Simulation)      = getshortname(sim) * '_' * sim.id


getshortname(obs::Observable)        = split("$obs", '{')[1]
getshortname(c::SimulationComponent) = split("$c", '{')[1]


getbzbounds(sim::Simulation) = getbzbounds(sim.drivingfield, sim.grid.kgrid)

function checkbzbounds(sim::Simulation)
	bz = getbzbounds(sim)
	if isempty(bz)
		return
	elseif bz[1] > bz[2] || (sim.dimensions == 2 && bz[3] > bz[4])
		@warn "Brillouin zone vanishes: $(bz)"
	end
end

function resize_obs!(sim::Simulation)
	sim.observables .= [resize(o, sim) for o in sim.observables]
end
resize(o::Observable, sim::Simulation) = resize(o, sim.grid)

function define_functions(sim::Simulation, solver::DamysosSolver)
	!solver_compatible(sim, solver) && throw(incompatible_solver_exception(sim, solver))
	return (
		define_rhs_x(sim, solver),
		define_bzmask(sim, solver),
		define_observable_functions(sim, solver))
end

function incompatible_solver_exception(sim::Simulation, solver::DamysosSolver)
	return ErrorException("""
		Solver $solver is incompatible with simulation. Compatible pairs are:
			LinearChunked => 1d & 2d Simulation
			LinearCUDA    => 1d & 2d Simulation
			SingleMode    => 0d Simulation
		Your Simulation has the dimension $(sim.dimensions)""")
end

printdimless_params(l::Liouvillian, df::DrivingField) = ""

function printparamsSI(sim::Simulation; digits = 3)
    
    us  = sim.unitscaling
    l   = sim.liouvillian
    df  = sim.drivingfield
    g   = sim.grid
	str = printdimless_params(l, df)

	str *= printBZSI(df, g.kgrid, us, digits = digits)
	str *= printparamsSI(l, us; digits = digits)
	str *= printparamsSI(df, us; digits = digits)
	str *= printparamsSI(g, us; digits = digits)
	return str
end



function markdown_paramsSI(sim::Simulation)

	input_str = printparamsSI(sim)
	table_str = "| Parameter | Value (SI units) | Value (scaled) |\n" *
				"|-----------|------------------|----------------|\n"

	# Split the input string into lines
	lines = split(input_str, '\n')

	for line in lines
		# Use regular expressions to extract values
		pattern = r"(.+?)\s*=\s*([^()]+)\s*(?:\(([\d\.]+)\))?"

		# Match the pattern in the input string
		match_result = match(pattern, line)

		if match_result !== nothing
			# Extract matched groups
			parameter_name = match_result[1]
			first_number = match_result[2]
			number_in_brackets = isnothing(match_result[3]) ? " " : match_result[3]

			# Append a new row to the table string
			table_str *= "| $parameter_name | $first_number | $number_in_brackets |\n"
		end
	end

	return table_str
end

