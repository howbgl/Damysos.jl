
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
		kchunksize::T = default_kchunk_size(LinearCUDA),
		algorithm::DiffEqGPU.GPUODEAlgorithm = GPUTsit5()) where {T}

		if !CUDA.functional()
			@warn "CUDA.jl is not functional, cannot use LinearCUDA solver."
		end
		return new(kchunksize, algorithm)
	end
end


function LinearCUDA(
	kchunksize::Integer = default_kchunk_size(LinearCUDA),
	algorithm::DiffEqGPU.GPUODEAlgorithm = GPUVern7())
	return LinearCUDA{typeof(kchunksize)}(kchunksize, algorithm)
end

default_kchunk_size(::Type{LinearCUDA}) = 10_000

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


function _run!(
	sim::Simulation,
	functions,
	solver::LinearCUDA;
	bypass_memcheck = false)

	!CUDA.functional() && throw(ErrorException(
		"CUDA.jl is not functional, cannot use LinearCUDA solver."))

	obs_kchunks = Vector{Vector{Observable}}(undef, 0)
	kchunks 	= buildkgrid_chunks(sim,solver.kchunksize)

	ts, obs = ([gettsamples(sim)], [sim.observables])
	if !bypass_memcheck
		actual_n_kchunk = minimum((solver.kchunksize,length(kchunks[1])))
		ts, obs = build_tchunks_if_necessary(sim, functions, solver, actual_n_kchunk)
	end


	@progress name = "Simulation" for ks in kchunks
		obs_timeslice = deepcopy(obs)
		for (t, o) in zip(ts, obs_timeslice)
			runkchunk!(sim, functions, solver, ks, t, o)
		end
		push!(obs_kchunks, timemerge_obs(obs_timeslice))
	end

	sim.observables .= sum(obs_kchunks)
	CUDA.reclaim()

	return sim.observables
end

function runkchunk!(
	sim::Simulation,
	functions,
	solver::LinearCUDA,
	kchunk::Vector{<:SVector{2, <:Real}},
	ts::AbstractVector{<:Real} = gettsamples(sim),
	obs::Vector{<:Observable} = sim.observables)

	rhs, bzmask, obsfuncs = functions
	d_ts, d_us = solvechunk(sim, solver, kchunk, rhs, ts)

	sum_observables!(cu(kchunk), d_us, d_ts, bzmask, obs, obsfuncs)

	# it seems that sometimes the GC is too slow, so to be safe free GPU arrays
	CUDA.unsafe_free!(d_ts)
	CUDA.unsafe_free!(d_us)

	return obs
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
	tsamples::AbstractVector{T} = collect(gettsamples(sim))) where {T <: Real}

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
	obsfuncs::Vector{<:Vector};
	free_memory = true)

	@debug "Summing observables"

	@debug "Allocating weigths"
	d_weigths = bzmask.(d_kchunk', d_ts)
	@debug "Allocating buffer"
	buf     = zero(d_ts)
	obsnfns = zip(obs, obsfuncs)
	@debug "Summing"
	res = [sum_observables!(o, f, d_kchunk, d_us, d_ts, d_weigths, buf) for (o, f) in obsnfns]

	if free_memory
		CUDA.unsafe_free!(d_weigths)
		CUDA.unsafe_free!(buf)
	end

	@debug "Finished summing observables"
	return res
end

function build_tchunks_if_necessary(sim::Simulation, fns, solver::LinearCUDA, nk::Integer)

	obs = deepcopy(sim.observables)
	ts = [gettsamples(sim)]
	mem_est = cuda_memory_estimate_linear(sim, fns, solver)
	nt = getnt(sim)
	estimate = nk * mem_est(nt)

	@info "Memory estimate: $(100*estimate) %"

	if nk * mem_est(nt) > 0.9
		nt = adjust_tsamples_memory(getnt(sim), nk, mem_est)
		ts = subdivide_vector(gettsamples(sim), nt)
		obs = timesplit_obs(obs, ts)
		@warn """
		Memory estimate > available mem of CUDA context. Subdividing tsamples into
		$(length(ts)) timeslices. This may impact runtime negatively. If sensible, try to 
		choose a smaller kchunksize (=$nk) in the LinearCUDA solver instead."""
		return ts, obs
	else
		return ts, [obs]
	end
end

function cuda_memory_estimate_linear(sim::Simulation, fns, solver::LinearCUDA,
	nkchunk::Integer = 8_000)

	@info "Running small simulation test-chunk to estimate memory consumption"

	mem_fractions = Float64[]

	GC.gc(true)
	CUDA.reclaim()
	maxnt = minimum((getnt(sim),5_000))
	nts = div.(maxnt,4:-1:1)

	for nt in nts
		kchunk = buildkgrid_chunks(sim, nkchunk)[1]
		nk = length(kchunk) # account for possible nk < nkchunk
		obs = deepcopy(sim.observables)
		ts = subdivide_vector(gettsamples(sim), nt)
		obs = timesplit_obs(obs, ts)[1]

		rhs, bzmask, obsfuncs = fns
		d_ts, d_us = solvechunk(sim, solver, kchunk, rhs, ts[1])
		sum_observables!(cu(kchunk), d_us, d_ts, bzmask, obs, obsfuncs; free_memory = false)

		free, tot = CUDA.memory_info()
		push!(mem_fractions, (tot - free) / (tot * nk) )

		CUDA.unsafe_free!(d_ts)
		CUDA.unsafe_free!(d_us)
		GC.gc(true)
		CUDA.reclaim()

	end

	return linear_fit(nts,mem_fractions)
end


function adjust_tsamples_memory(nt::Integer, nkchunk::Integer, memory_est)
	current_nt = nt
	while memory_est(current_nt) * nkchunk > 0.9
		current_nt -= 10
		current_nt < 10 && throw(ErrorException(
			"Simulation needs extremly small time-chunks (<10)!"))
	end
	return current_nt
end

bytes_to_gb(x::Real) = x / 1073741824
