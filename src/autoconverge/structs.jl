export ConvergenceTest
export ConvergenceTestMethod
export ConvergenceTestResult
export CTestStart
export LinearTest
export PowerLawTest

abstract type ConvergenceTestMethod end

@enumx ReturnCode success maxtime maxiter running failed exception

"""
    ConvergenceTest(start, solver::DamysosSolver = LinearChunked(); kwargs...)

A convergence test based on a Simulation and a ConvergenceTestMethod.

# Arguments
- `start`: the starting point can be a `Simulation` object or are path to a previous test
- `solver::DamysosSolver = LinearChunked()`: the solver used for simulations,

# Keyword arguments
- `method::ConvergenceTestMethod`: specifies the convergence parmeter & iteration method
- `resume::Bool`: if true re-use the completedsims, otherwise start from scratch
- `atolgoal::Real`: desired absolute tolerance
- `rtolgoal::Real`: desired relative tolerance
- `maxtime::Union{Real,Unitful.Time}`: test guaranteed to stop after maxtime
- `maxiterations::Integer`: test stops after maxiterations Simulations were performed
- `path::String`: path to save data of convergence test
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
		solver::DamysosSolver = LinearChunked();
		method::ConvergenceTestMethod = PowerLawTest(:dt, 0.5),
		atolgoal::Real = 1e-12,
		rtolgoal::Real = 1e-8,
		maxtime::Union{Real, Unitful.Time} = 600,
		maxiterations::Integer = 16,
		path::String = joinpath(start.datapath, "convergencetest_$(getname(method)).hdf5"),
		completedsims::Vector{<:Simulation} = empty([start]),
		resume = false,
		altpath = joinpath(
			pwd(), 
			"convergencetest_$(basename(tempname()))_$(getname(method)).hdf5"))

		maxtime = maxtime isa Real ? maxtime : ustrip(u"s", maxtime)

		(success, path) = ensurefilepath([path, altpath])
		!success && throw(ErrorException("could not create neceesary data directory"))
		
		@reset start.datapath = joinpath(path, "start")
		@reset start.plotpath = joinpath(path, "start")
		@reset start.id = "#1"

		!resume && rename_file_if_exists(path)

		fns 	= []
		s 		= deepcopy(start)

		maxiterations = maxiterations + length(completedsims)

		for i in 1:maxiterations
			f = define_functions(s, solver)
			@debug """
				Defining functions (iteration $i)
					$(s.numericalparams)
					$(s.drivingfield)
					$(s.liouvillian)
					
					
					"""
			s = next(s, method)
			push!(fns, f)
		end

		return new(
			start,
			solver,
			method,
			atolgoal,
			rtolgoal,
			maxtime,
			maxiterations,
			completedsims,
			path,
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
	extrapolated_results::Vector{<:Observable}
end


struct ObservableExtrapolation{T <: Real}
	observables::Vector{Observable{T}}
	errs::Vector{<:Vector{T}}
end