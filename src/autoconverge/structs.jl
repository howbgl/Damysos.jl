export ConvergenceTest
export ConvergenceTestMethod
export ConvergenceTestResult
export LinearTest
export PowerLawTest

abstract type ConvergenceTestMethod end

@enumx ReturnCode success maxtime maxiter running failed

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

		rename_file_if_exists(filepath)
		h5open(filepath,"cw") do file
			create_group(file,"completedsims")
			savedata_hdf5(method,file)
			file["atolgoal"] 		= atolgoal
			file["rtolgoal"] 		= rtolgoal
			file["maxtime"]  		= maxtime
			file["maxiterations"] 	= maxiterations
			file["testdatafile"] 	= filepath
		end

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

struct LinearTest{T <: Real} <: ConvergenceTestMethod
	parameter::Symbol
	shift::T
end

struct PowerLawTest{T <: Real} <: ConvergenceTestMethod
	parameter::Symbol
	multiplier::T
end


struct ConvergenceTestResult
	test::ConvergenceTest
	retcode::ReturnCode.T
	min_achieved_atol::Real
	min_achieved_rtol::Real
	elapsed_time_sec::Real
	iterations::Integer
	last_params::NumericalParameters
end
