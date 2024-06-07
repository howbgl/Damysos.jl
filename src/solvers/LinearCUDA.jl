
export LinearCUDA

"""
	LinearCUDA{T}

Represents an integration strategy for k-space via simple midpoint sum, where individual
k points are computed concurrently on a CUDA GPU

# Fields
- `kchunksize::T`: number of k-points in one concurrently executed chunk. 
- `algorithm::GPUODEAlgorithm`: algorithm for solving differential equations

# See also
[`LinearChunked`](@ref LinearChunked), [`SingleMode`](@ref SingleMode)
"""
struct LinearCUDA{T <: Integer} <: DamysosSolver
	kchunksize::T
	algorithm::DiffEqGPU.GPUODEAlgorithm
	function LinearCUDA{T}(
		kchunksize::T = DEFAULT_K_CHUNK_SIZE,
		algorithm::DiffEqGPU.GPUODEAlgorithm = GPUTsit5()) where {T}

		if CUDA.functional()
			new(kchunksize, algorithm)
		else
			throw(ErrorException(
				"CUDA.jl is not functional, cannot use LinearCUDA solver."))
		end
	end
end


function LinearCUDA(
	kchunksize::Integer = DEFAULT_K_CHUNK_SIZE,
	algorithm::DiffEqGPU.GPUODEAlgorithm = GPUVern7())
	return LinearCUDA{typeof(kchunksize)}(kchunksize, algorithm)
end

function solver_compatible(sim::Simulation, ::LinearCUDA)
	return sim.dimensions == 2 || sim.dimensions == 1
end

function Base.show(io::IO, ::MIME"text/plain", s::LinearCUDA)
	println(io, "LinearCUDA:" |> escape_underscores)
	str = """
	- kchunksize: $(s.kchunksize)
	- algorithm: $(s.algorithm)
	""" |> escape_underscores
	print(io, prepend_spaces(str, 2))
end


function run!(
	sim::Simulation,
	functions,
	solver::LinearCUDA;
	savedata = true,
	saveplots = true)

	prerun!(sim, solver; savedata = savedata, saveplots = saveplots)

	@info """
		Solver: $(repr(solver))
	"""
	totalobs = Vector{Vector{Observable}}(undef, 0)

	@progress name = "Simulation" for ks in buildkgrid_chunks(sim, solver.kchunksize)

		obs = runkchunk!(sim, functions, solver, ks)
		push!(totalobs, obs)
	end

	sim.observables .= sum(totalobs)

	postrun!(sim; savedata = savedata, saveplots = saveplots)

	return sim.observables
end

function runkchunk!(
	sim::Simulation,
	functions,
	solver::LinearCUDA,
	kchunk::Vector{<:SVector{2, <:Real}})

	rhs, bzmask, obsfuncs = functions
    ts, obs  = build_tchunks_if_necessary(sim,solver,length(kchunk))

	for (t, o) in zip(ts, obs)
		d_ts, d_us = solvechunk(sim, solver, kchunk, rhs, t)

		sum_observables!(cu(kchunk), d_us, d_ts, bzmask, o, obsfuncs)

		# it seems that sometimes, the GC is too slow, so to be safe free GPU arrays
		CUDA.unsafe_free!(d_ts)
		CUDA.unsafe_free!(d_us)
	end

	return timemerge_obs(obs)
end

define_rhs_x(sim::Simulation, ::LinearCUDA) = @eval (u, p, t) -> $(buildrhs_x_expression(sim))

define_bzmask(sim::Simulation, ::LinearCUDA) = define_bzmask(sim)

function define_observable_functions(sim::Simulation, solver::LinearCUDA)
	return [define_observable_functions(sim, solver, o) for o in sim.observables]
end

function define_observable_functions(sim::Simulation, ::LinearCUDA, o::Observable)
	return [@eval (u, p, t) -> $ex for ex in buildobservable_vec_of_expr(sim, o)]
end

function solvechunk(
	sim::Simulation{T},
	solver::LinearCUDA,
	kchunk::Vector{SVector{2, T}},
	rhs::Function,
	tsamples::Vector{T} = collect(gettsamples(sim))) where {T <: Real}

	prob = ODEProblem{false}(
		rhs,
		SA[zeros(Complex{T}, 2)...],
		gettspan(sim),
		kchunk[1])

	probs = map(kchunk) do k
		DiffEqGPU.make_prob_compatible(remake(prob, p = k))
	end
	probs = cu(probs)
	CUDA.@sync ts, us = DiffEqGPU.vectorized_solve(
		probs,
		prob,
		solver.algorithm;
		save_everystep = false,
		saveat = tsamples,
		dt = sim.numericalparams.dt)

	return ts, us
end

function sum_observables!(
	d_kchunk::CuArray{<:SVector{2, <:Real}},
	d_us::CuArray{<:SVector{2, <:Complex}},
	d_ts::CuArray{<:Real, 2},
	bzmask::Function,
	obs::Vector{<:Observable},
	obsfuncs::Vector{<:Vector})

	d_weigths = bzmask.(d_kchunk', d_ts)
	buf       = zero(d_ts)
	obsnfns   = zip(obs, obsfuncs)
	return [sum_observables!(o, f, d_kchunk, d_us, d_ts, d_weigths, buf) for (o, f) in obsnfns]
end

function build_tchunks_if_necessary(sim::Simulation, solver::LinearCUDA, nk::Integer)

	obs          = deepcopy(sim.observables)
	ts           = [gettsamples(sim)]
	cuda_ctx_mem = CUDA.total_memory()

	if memory_estimate(sim, solver) > cuda_ctx_mem
		nt = adjust_tsamples_memory(getnt(sim), nk, cuda_ctx_mem)
		ts = subdivide_vector(gettsamples(sim), nt)
		obs = timesplit_obs(obs, ts)
		@warn """
        Memory estimate > available mem of CUDA context ($(bytes_to_gb(cuda_ctx_mem)) GB).
        Subdividing tsamples into $(length(ts)) timeslices. 
        This may impact runtime negatively. If sensible, try to choose a smaller
        kchunksize (=$nk) in the LinearCUDA solver instead."""
	end
	return ts, obs
end

function memory_estimate(sim::Simulation, solver::LinearCUDA)
	return memory_estimate(getnt(sim.numericalparams), solver.kchunksize)
end

function memory_estimate(nt::Integer, nkchunk::Integer)
	# The raw data takes about 5 * kchunksize * nt * 8 bytes, but allow for some
	# wiggle room for the solver etc.
	return 6 * nt * nkchunk * 8
end

function adjust_tsamples_memory(nt::Integer, nkchunk::Integer, memory_available::Integer)
	if memory_estimate(nt, nkchunk) < memory_available
		return nt
	else
		return adjust_tsamples_memory(div(nt, 2), nkchunk, memory_available)
	end
end

bytes_to_gb(x::Real) = x / 1073741824
