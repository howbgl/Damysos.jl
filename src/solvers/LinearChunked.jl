
export LinearChunked

const DEFAULT_ATOL = Float64(1e-12)
const DEFAULT_RTOL = Float64(1e-6)

"""
	LinearChunked

Represents an integration strategy for k-space via simple midpoint sum.

# Fields
- `kchunksize::T`: number of k-points in one chunk. Every task/worker gets one chunk. 
- `algorithm::SciMLBase.BasicEnsembleAlgorithm`: algorithm for the `EnsembleProblem`.
- `odesolver::SciMLBase.AbstractODEAlgorithm`: ODE algorithm

# Examples
```jldoctest
julia> solver = LinearChunked(256,EnsembleThreads())
LinearChunked:
  - kchunksize: 256
  - algorithm: EnsembleThreads()
  - odesolver: Vern7{typeof(OrdinaryDiffEqCore.trivial_limiter!), typeof(OrdinaryDiffEqCore.trivial_limiter!), Static.False}(OrdinaryDiffEqCore.trivial_limiter!, OrdinaryDiffEqCore.trivial_limiter!, static(false), true)
  
```

# See also
[`LinearChunked`](@ref LinearChunked), [`SingleMode`](@ref SingleMode)
"""
struct LinearChunked <: DamysosSolver
	kchunksize::Int64
	algorithm::SciMLBase.BasicEnsembleAlgorithm
	odesolver::SciMLBase.AbstractODEAlgorithm
	atol::Float64
	rtol::Float64
end
function LinearChunked(
	kchunksize::Integer = default_kchunk_size(LinearChunked),
	algorithm = choose_threaded_or_distributed(),
	odesolver = Vern7(),
	atol = DEFAULT_ATOL,
	rtol = DEFAULT_RTOL)
	LinearChunked(kchunksize, algorithm, odesolver, atol, rtol)
end

default_kchunk_size(::Type{LinearChunked}) = Int64(256)

function solver_compatible(sim::Simulation, ::LinearChunked)
	return sim.dimensions == 2 || sim.dimensions == 1
end

function _run!(
	sim::Simulation,
	functions,
	solver::LinearChunked)

	rhscc, rhscv = functions[1]
	fns = (rhscc, rhscv, functions[2:end]...)

	prob, kchunks = buildensemble(sim, solver, fns...)

	# At DifferentialEquations.jl > 7.10 auto-detection of ode alg throws error due to
	# ForwardDiff of ComplexF64, so use ode_alg workaround
	ode_alg = AutoVern7(KenCarp47(autodiff = false), lazy = true)

	res = solve(
		prob,
		solver.odesolver,
		solver.algorithm;
		trajectories = length(kchunks),
		saveat = gettsamples(sim),
		abstol = solver.atol,
		reltol = solver.rtol,
		progress = true)

	return sim.observables
end

function define_rhs_x(sim::Simulation, ::LinearChunked)

	ccex, cvex = buildrhs_cc_cv_x_expression(sim)
	return (@eval (cc, cv, kx, ky, t) -> $ccex, @eval (cc, cv, kx, ky, t) -> $cvex)
end

define_bzmask(sim::Simulation, ::LinearChunked) = define_bzmask(sim)

function define_observable_functions(sim::Simulation, ::LinearChunked)
	return [define_observable_functions(sim, LinearChunked(), o) for o in sim.observables]
end

function define_observable_functions(sim::Simulation, ::LinearChunked, o::Observable)
	return [@eval (cc, cv, p, t) -> $ex for ex in buildobservable_vec_of_expr_cc_cv(sim, o)]
end

function observables_out(sol, bzmask, obsfunctions, observables)

	p       = sol.prob.p
	weigths = bzmask.(p, sol.t')
	cc      = sol[1:2:end,:]
	cv      = sol[2:2:end,:]
	obs     = zero.(observables)

	for (fns, o) in zip(obsfunctions, obs)
		sum_observables!(o, fns, p, cc, cv,sol.t, weigths)
	end

	return (obs, false)
end


function buildensemble(
	sim::Simulation,
	solver::LinearChunked,
	rhs_cc::Function,
	rhs_cv::Function,
	bzmask::Function,
	obsfunctions)

	kbatches = buildkgrid_chunks(sim.grid.kgrid, solver.kchunksize)
	prob     = buildode(sim, solver, kbatches[1], rhs_cc, rhs_cv)
	ensprob  = EnsembleProblem(
		prob,
		prob_func   = let kb = kbatches
			(prob, i, repeat) -> remake(
			prob,
			p = kb[i],
			u0 = zeros(Complex{eltype(kb[i][1])}, 2length(kb[i])))
		end,
		output_func = (sol, i) -> observables_out(sol, bzmask, obsfunctions, sim.observables),
		reduction   = (u, data, I) -> begin 
			u .+= sum(data)
			return 	(u, false)
		end,
		u_init = sim.observables,
		safetycopy  = false)

	return ensprob, kbatches
end

function buildode(
	sim::Simulation{T},
	::LinearChunked,
	kbatch::Vector{SVector{2, T}},
	rhs_cc::Function,
	rhs_cv::Function) where {T <: Real}

	tspan = gettspan(sim)
	u0    = zeros(Complex{T}, 2length(kbatch))

	function f(du, u, p, t)
		for i in 1:length(p)
			@inbounds du[2i-1] = rhs_cc(u[2i-1], u[2i], p[i][1], p[i][2], t)
			@inbounds du[2i]   = rhs_cv(u[2i-1], u[2i], p[i][1], p[i][2], t)
		end
	end

	return ODEProblem{true}(f, u0, tspan, kbatch)
end


function choose_threaded_or_distributed()

	nthreads = Threads.nthreads()
	nworkers = Distributed.nworkers()

	if nthreads == 1 && nworkers == 1
		return EnsembleSerial()
	elseif nthreads > 1 && nworkers == 1
		return EnsembleThreads()
	elseif Threads.nthreads() == 1 && nworkers > 1
		return EnsembleDistributed()
	else
		@warn """"
		Multiple threads and processes detected. This might result in unexpected behavior.
		Using EnsembleDistributed() nonetheless."""
		return EnsembleDistributed()
	end
end

getserialsolver(solver::LinearChunked) = LinearChunked(solver.kchunksize, EnsembleSerial())

function Base.show(io::IO, ::MIME"text/plain", s::LinearChunked)
	println(io, "LinearChunked:")
	str = """
	- kchunksize: $(s.kchunksize)
	- algorithm: $(s.algorithm)
	- odesolver: $(s.odesolver)
	"""
	print(io, prepend_spaces(str, 2))
end

