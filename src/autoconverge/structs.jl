export ConvergenceTest
export ConvergenceTestMethod
export ConvergenceTestResult
export LinearTest
export PowerLawTest

abstract type ConvergenceTestMethod end

@enumx ReturnCode success maxtime maxiter running failed


"""
    ConvergenceTest(start::Simulation,
		solver::DamysosSolver = LinearChunked(),
		method::ConvergenceTestMethod = PowerLawTest(:dt, 0.5),
		atolgoal::Real = 1e-12,
		rtolgoal::Real = 1e-8,
		maxtime::Union{Real, Unitful.Time} = 600,
		maxiterations::Integer = 16;
		altpath = joinpath(pwd(), start.datapath)))

A convergence test based on a Simulation and a ConvergenceTestMethod.

# Arguments
- `start::Simulation`: the Simulation to be converged
- `method::ConvergenceTestMethod`: specifies the convergence parmeter & iteration method
- `atolgoal::Real`: desired absolute tolerance
- `rtolgoal::Real`: desired relative tolerance
- `maxtime::Union{Real,Unitful.Time}`: test guaranteed to stop after maxtime
- `maxiterations::Integer`: test stops after maxiterations Simulations were performed
- `altpath`: path to try inf start.datapath throws an error


# See also
[`LinearTest`](@ref), [`PowerLawTest`](@ref), [`Simulation`](@ref)
"""
struct ConvergenceTest
	start::Simulation
	solver::DamysosSolver
	method::ConvergenceTestMethod
	atolgoal::Real
	rtolgoal::Real
	maxtime::Real
	maxiterations::Integer
	completedsims::Vector{Simulation}
	testdatafile::String
	allfunctions::Vector{Any}
	function ConvergenceTest(
		start::Simulation,
		solver::DamysosSolver = LinearChunked(),
		method::ConvergenceTestMethod = PowerLawTest(:dt, 0.5),
		atolgoal::Real = 1e-12,
		rtolgoal::Real = 1e-8,
		maxtime::Union{Real, Unitful.Time} = 600,
		maxiterations::Integer = 16;
		altpath = joinpath(pwd(), start.datapath))

		maxtime = maxtime isa Real ? maxtime : ustrip(u"s", maxtime)

		fns = []
		s = deepcopy(start)

		for i in 1:maxiterations
			f = define_functions(s, solver)
			s = next(s, method)
			push!(fns, f)
		end

		(success, path) = ensurepath([start.datapath, altpath])
		!success && throw(ErrorException("could not create neceesary data directory"))

		filename = "convergencetest_$(getname(start))_$(getname(method)).hdf5"
		filepath = joinpath(path, filename)
		
		@reset start.datapath = joinpath(start.datapath, "start")
		@reset start.plotpath = joinpath(start.plotpath, "start")
		@reset start.id = "start_$(start.id)"

		return new(
			start,
			solver,
			method,
			atolgoal,
			rtolgoal,
			maxtime,
			maxiterations,
			empty([start]),
			filepath,
			fns)
	end
end

"""
    LinearTest{T<:Real}(parameter::Symbol,shift{T})

A convergence method where `parameter` is changed by adding `shift` each iteration.

# See also
[`ConvergenceTest`](@ref), [`PowerLawTest`](@ref)
"""
struct LinearTest{T <: Real} <: ConvergenceTestMethod
	parameter::Symbol
	shift::T
end

"""
    PowerLawTest{T<:Real}(parameter::Symbol,multiplier{T})

A convergence method multiplying `parameter` by `multiplier` each iteration.

# See also
[`ConvergenceTest`](@ref), [`LinearTest`](@ref)
"""
struct PowerLawTest{T <: Real} <: ConvergenceTestMethod
	parameter::Symbol
	multiplier::T
end

"Result of a ConvergenceTest"
struct ConvergenceTestResult
	test::ConvergenceTest
	retcode::ReturnCode.T
	min_achieved_atol::Real
	min_achieved_rtol::Real
	elapsed_time_sec::Real
	iterations::Integer
	last_params::NumericalParameters
end
