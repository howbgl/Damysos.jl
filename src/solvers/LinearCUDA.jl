
export LinearCUDA

"""
	LinearCUDA

Represents an integration strategy for k-space via simple midpoint sum, where individual
k points are computed concurrently on one or several CUDA GPU(s) via linear indexing.

# Fields
- `kchunksize::Int64`: number of k-points in one concurrently executed chunk. 
- `algorithm::GPUODEAlgorithm`: algorithm for solving differential equations
- `ngpus::Int`: #GPUs to use, automatically chooses all available GPUs if none given
- `rtol::Union{Nothing, Real}`: relative tolerance of solver (nothing for fixed-timestep)
- `atol::Union{Nothing, Real}`: absolute tolerance of solver (nothing for fixed-timestep)

# See also
[`LinearChunked`](@ref LinearChunked), [`SingleMode`](@ref SingleMode)
"""
struct LinearCUDA{isadaptive} <: DamysosSolver
	kchunksize::Int64
	algorithm::DiffEqGPU.GPUODEAlgorithm
	ngpus::Int64
	rtol::Real
	atol::Real
	function LinearCUDA(
		kchunksize::Int64 = default_kchunk_size(LinearCUDA),
		algorithm::DiffEqGPU.GPUODEAlgorithm = GPUVern7(),
		ngpus::Int64 = convert(Int64, CUDA.functional() ? length(CUDA.devices()) : 1);
		rtol = nothing,
		atol = nothing)

		adaptive 	= !(isnothing(rtol) && isnothing(atol))
		_rtol 		= isnothing(rtol) ? DEFAULT_RTOL : rtol
		_atol 		= isnothing(atol) ? DEFAULT_ATOL : atol

		if CUDA.functional()
			_ngpus = length(CUDA.devices())
			if _ngpus < ngpus
				@warn """
					Only $(_ngpus) GPUs available (requested $ngpus)
					Proceeding with $(_ngpus) GPUs"""
			elseif _ngpus > ngpus
				@warn """
					Only requested $ngpus out of the $(_ngpus) available GPUs.
					Proceeding with $(ngpus) GPUs"""
				_ngpus = ngpus
			end
			return new{adaptive}(kchunksize, algorithm, _ngpus, _rtol, _atol)
		else
			@warn "CUDA.jl is not functional, cannot use LinearCUDA solver."
			return new{adaptive}(kchunksize, algorithm, ngpus, _rtol, _atol)
		end
	end
end

default_kchunk_size(::Type{LinearCUDA}) = Int64(10_000)

function solver_compatible(sim::Simulation, ::LinearCUDA)
	return sim.dimensions == 2 || sim.dimensions == 1
end

function Base.show(io::IO, ::MIME"text/plain", s::LinearCUDA{ia}) where {ia}
	println(io, "LinearCUDA{$ia}:" |> escape_underscores)
	str = """
	- kchunksize: $(s.kchunksize)
	- algorithm: $(s.algorithm)
	- # GPUs: $(s.ngpus)""" 
	if ia
		str *= """
		- rtol: $(s.rtol)
		- atol: $(s.atol)"""
	end
	print(io, prepend_spaces(escape_underscores(str), 2))
end


function _run!(
		sim::Simulation,
		functions,
		solver::LinearCUDA;
		bypass_memcheck = false) 
		
	kchunks = buildkgrid_chunks(sim.grid.kgrid, solver.kchunksize)
	kchunk_batches = subdivide_vector(kchunks, cld(length(kchunks), solver.ngpus))

	synchronize()

	function work(d, kcs)
		device!(d)
		res = runtimeslices!(sim, functions, solver, kcs; bypass_memcheck = bypass_memcheck)
		CUDA.reclaim()
		return res
	end

	obs_batches = asyncmap(work, devices(), kchunk_batches)
	sim.observables .= sum(obs_batches)

	return sim.observables
end

function runtimeslices!(
		sim::Simulation,
		functions,
		solver::LinearCUDA,
		kchunks;
		bypass_memcheck = false)

	!CUDA.functional() && throw(ErrorException(
		"CUDA.jl is not functional, cannot use LinearCUDA solver."))

	obs_kchunks = Vector{Vector{Observable}}(undef, 0)
	ts, obs = ([gettsamples(sim)], [sim.observables])

	if !bypass_memcheck
		actual_n_kchunk = minimum((solver.kchunksize, length(kchunks[1])))
		ts, obs = build_tchunks_if_necessary(sim, functions, solver, actual_n_kchunk)
	end


	@progress name = "Simulation" for ks in kchunks
		obs_timeslice = deepcopy(obs)
		for (t, o) in zip(ts, obs_timeslice)
			runkchunk!(sim, functions, solver, ks, t, o)
		end
		push!(obs_kchunks, timemerge_obs(obs_timeslice))
	end

	return sum(obs_kchunks)
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

define_rhs_x(sim::Simulation, ::LinearCUDA) = @eval (u, p, t) -> $(buildrhs_expression_svec(sim))

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
	return solve_ode_problems(probs, prob, solver, tsamples, getdt(sim))
end

function solve_ode_problems(problems::CuArray, prob, solver::LinearCUDA{false}, saveat, dt)
	
	CUDA.@sync ts, us = DiffEqGPU.vectorized_solve(
		problems,
		prob,
		solver.algorithm;
		save_everystep = false,
		saveat = saveat,
		dt = dt)

	return ts, us
end

function solve_ode_problems(problems::CuArray, prob, solver::LinearCUDA{true}, saveat, dt)
	
	CUDA.@sync ts, us = DiffEqGPU.vectorized_asolve(
		problems,
		prob,
		solver.algorithm;
		save_everystep = false,
		saveat = saveat,
		dt = dt,
		abstol = solver.atol,
		reltol = solver.rtol)

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
	maxnt = minimum((getnt(sim), 5_000))
	nts = div.(maxnt, 4:-1:1)

	for nt in nts
		kchunk = buildkgrid_chunks(sim.grid.kgrid, nkchunk)[1]
		nk = length(kchunk) # account for possible nk < nkchunk
		obs = deepcopy(sim.observables)
		ts = subdivide_vector(gettsamples(sim), nt)
		obs = timesplit_obs(obs, ts)[1]

		rhs, bzmask, obsfuncs = fns
		d_ts, d_us = solvechunk(sim, solver, kchunk, rhs, ts[1])
		sum_observables!(cu(kchunk), d_us, d_ts, bzmask, obs, obsfuncs; free_memory = false)

		free, tot = CUDA.memory_info()
		push!(mem_fractions, (tot - free) / (tot * nk))

		CUDA.unsafe_free!(d_ts)
		CUDA.unsafe_free!(d_us)
		GC.gc(true)
		CUDA.reclaim()

	end

	return linear_fit(nts, mem_fractions)
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
