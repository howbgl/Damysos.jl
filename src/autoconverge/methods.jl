export successful_retcode
export terminated_retcode
export resume


function ConvergenceTest(
	filepath_hdf5::String,
	solver::DamysosSolver = LinearChunked();
	kwargs...)
	h5open(filepath_hdf5,"r") do file
		return ConvergenceTest(file,solver;kwargs...)
	end
end

function ConvergenceTest(
	file::Union{HDF5.File, HDF5.Group},
	solver::DamysosSolver = LinearChunked();
	method::ConvergenceTestMethod = load_obj_hdf5(file["method"]),
	atolgoal::Real = read(file,"atolgoal"),
	rtolgoal::Real = read(file,"rtolgoal"),
	maxtime::Union{Real, Unitful.Time} = read(file,"maxtime"),
	maxiterations::Integer = read(file,"maxiterations"),
	path::String = read(file["testdatafile"]),
	altpath = joinpath(pwd(), read(file,"testdatafile")),
	resume = true)
	
	g 			= file["completedsims"]
	done_sims 	= [load_obj_hdf5(g[s]) for s in keys(g)]
	sort!(done_sims,by=getsimindex)
	start		= resume ? load_obj_hdf5(file["start"]) : last(done_sims)
	
	return ConvergenceTest(
		start,
		solver;
		method=method,
		atolgoal=atolgoal,
		rtolgoal=rtolgoal,
		maxtime=maxtime,
		maxiterations=maxiterations,
		path=path,
		completedsims=resume ? done_sims : empty([start]),
		altpath = altpath)
end

"""
    successful_retcode(x)

Returns true if x terminated with a successful return code.

"""
successful_retcode(ctr::ConvergenceTestResult) = successful_retcode(ctr.retcode)

function successful_retcode(retcode::ReturnCode.T)
	return retcode == ReturnCode.success
end


"""
    successful_retcode(path::String)

Loads .hdf5 file of convergence test and returns true if it was successful.

# See also
[`ConvergenceTest`](@ref)
"""
function successful_retcode(path::String)
	h5open(path, "r") do file
		return successful_retcode(ReturnCode.T(read(file["testresult"], "retcode")))
	end
end

"""
    terminated_retcode(x)

Returns true if x terminated regularly.

"""
function terminated_retcode(retcode::ReturnCode.T)
	return any(retcode .== [
		ReturnCode.success,
		ReturnCode.maxtime,
		ReturnCode.maxiter,
		ReturnCode.failed])
end

terminated_retcode(x::ConvergenceTestResult) = successful_retcode(x.retcode)
"""
    terminated_retcode(path::String)

Loads .hdf5 file of convergence test and returns true if it terminated regularly.

# See also
[`ConvergenceTest`](@ref)
"""
function terminated_retcode(path::String)
	h5open(path, "r") do file
		return terminated_retcode(ReturnCode.T(read(file["testresult"], "retcode")))
	end
end

nextvalue(oldvalue::Real, method::PowerLawTest) = method.multiplier * oldvalue
nextvalue(oldvalue::Real, method::LinearTest)   = oldvalue + method.shift

function getsimindex(sim::Simulation)
	m = match(r"(?<=#)\d+",sim.id)
	isnothing(m) && throw(ErrorException(
		"Could not extract simulation index from its id $(sim.id)"))
	return parse(Int,m.match)
end

function currentvalue(m::Union{PowerLawTest, LinearTest}, sim::Simulation)
	return getproperty(sim.numericalparams, m.parameter)
end

function getfilename(m::Union{PowerLawTest, LinearTest}, sim::Simulation)
	return "$(m.parameter)=$(currentvalue(m,sim))_$(round(now(),Dates.Second))"
end


getname(t::ConvergenceTest) = "convergencetest_$(getname(t.start))_$(getname(t.method))"
getname(m::PowerLawTest)    = "PowerLawTest_$(m.parameter)"
getname(m::LinearTest)      = "LinearTest_$(m.parameter)"

function next(
	sim::Simulation,
	method::Union{PowerLawTest, LinearTest},
	parentdatapath::String = droplast(sim.datapath),
	parentplotpath::String = droplast(sim.plotpath))

	oldparam = getproperty(sim.numericalparams, method.parameter)
	opt      = PropertyLens(method.parameter)
	newparam = nextvalue(oldparam, method)
	params   = set(deepcopy(sim.numericalparams), opt, newparam)
	id       = "#$(getsimindex(sim)+1)_$(method.parameter)=$newparam"

	Simulation(
		sim.liouvillian,
		sim.drivingfield,
		params,
		zero.(sim.observables),
		sim.unitscaling,
		id,
		joinpath(parentdatapath, id),
		joinpath(parentplotpath, id))
end

function run!(
	test::ConvergenceTest;
	savedata = true,
	saveplots = false)

	if savedata && isempty(test.completedsims)
		rename_file_if_exists(test.testdatafile)
		h5open(test.testdatafile,"w") do file
			ensuregroup(file,"completedsims")
			savedata_hdf5(test.method,file)
			file["atolgoal"] 		= test.atolgoal
			file["rtolgoal"] 		= test.rtolgoal
			file["maxtime"]  		= test.maxtime
			file["maxiterations"] 	= test.maxiterations
			file["testdatafile"] 	= test.testdatafile
			savedata_hdf5(test.start,create_group(file,"start"))
		end
	end

	@info """
	## Starting $(repr("text/plain", test))
	$(repr("text/plain", test.method))
	"""
	printinfo(test.start,test.solver)

	runtask = @async begin
		try
			_run!(test, test.method; savedata = savedata,saveplots = saveplots)
		catch e
			if e isa InterruptException
				@warn "Convergence test interrupted!"
				close(test.testdatafile)
			end
		end

	end

	pollinterval = minimum((60.0, test.maxtime / 10))
	stats =
		@timed timedwait(() -> istaskdone(runtask), test.maxtime, pollint = pollinterval)
	timedout = stats.value == :timed_out
	if timedout
		schedule(runtask, InterruptException(), error = true)
		@info "Waiting for 120s to make time for cleanup"
		sleep(120)
	end

	return postrun!(test, stats.time, timedout; savedata = savedata)
end

function _run!(
	test::ConvergenceTest,
	method::Union{PowerLawTest, LinearTest};
	savedata = true,
	saveplots = false)

	done_sims        = test.completedsims
	currentiteration = length(done_sims)

	while currentiteration < test.maxiterations

		currentiteration += 1
		currentsim = currentiteration == 1 ? test.start : next(done_sims[end], method)

		@info "Check index: $(currentiteration) ?= $(getsimindex(currentsim))"

		run!(currentsim,test.allfunctions[currentiteration],test.solver;
			showinfo=false,
			savedata=false,
			saveplots=saveplots)

		savedata && Damysos.savedata(test, currentsim)
		saveplots && Damysos.savedata(currentsim)
		
		push!(done_sims, currentsim)
		achieved_tol = findminimum_precision(test)
		@info """ 
			- Iteration $currentiteration of maximum of $(test.maxiterations)
			- Current atol: $(achieved_tol[1])
			- Current rtol: $(achieved_tol[2])
		"""
		converged(test) && break
	end
	return nothing
end

function postrun!(test::ConvergenceTest, elapsedtime_seconds::Real, timedout::Bool;
	savedata = true)

	if converged(test)
		@info """
		## Converged after $(round(elapsedtime_seconds/60,sigdigits=3))min and \
		$(length(test.completedsims)) iterations"""
		retcode = ReturnCode.success
	elseif timedout
		@info "Maximum runtime exceeded"
		retcode = ReturnCode.maxtime
	elseif length(test.completedsims) >= test.maxiterations
		@warn "Maximum number of iterations ($(test.maxiterations)) exceeded."
		retcode = ReturnCode.maxiter
	else
		@warn "Something very weird happened..."
		retcode = ReturnCode.failed
	end

	achieved_tol = findminimum_precision(test)

	last_params =
		isempty(test.completedsims) ? test.start.numericalparams :
		test.completedsims[end].numericalparams

	result = ConvergenceTestResult(
		test,
		retcode,
		achieved_tol...,
		elapsedtime_seconds,
		length(test.completedsims),
		last_params)

	savedata && Damysos.savedata(result)

	@info "$(repr("text/plain", result))"

	return result
end

function converged(test::ConvergenceTest)
	length(test.completedsims) < 2 ? false :
	isapprox(
		test.completedsims[end-1],
		test.completedsims[end];
		atol = test.atolgoal,
		rtol = test.rtolgoal)
end

function findminimum_precision(test::ConvergenceTest)
	return	length(test.completedsims) < 2 ? (Inf, Inf) : findminimum_precision(
			test.completedsims[end-1],
			test.completedsims[end])
end

function findminimum_precision(
	s1::Simulation,
	s2::Simulation,
	atols::AbstractVector{<:Real},
	rtols::AbstractVector{<:Real})

	!isapprox(s1, s2; atol = atols[1], rtol = rtols[1]) && return (Inf, Inf)

	min_achieved_atol = atols[1]
	min_achieved_rtol = rtols[1]

	# First find the lowest atol, since that is usually less problematic
	for atol in atols
		if isapprox(s1, s2; atol = atol, rtol = rtols[1])
			min_achieved_atol = atol
		else
			break
		end
	end
	for rtol in rtols
		if isapprox(s1, s2; atol = min_achieved_atol, rtol = rtol)
			min_achieved_rtol = rtol
		else
			break
		end
	end

	return (min_achieved_atol, min_achieved_rtol)
end

function findminimum_precision(
	s1::Simulation,
	s2::Simulation;
	max_atol = 10.0,
	max_rtol = 10.0,
)

	p1 = getparams(s1)
	p2 = getparams(s2)

	min_possible_atol = maximum([p1.atol, p2.atol])
	min_possible_rtol = maximum([p1.rtol, p2.rtol])

	# Sweep the range of tolerance exponentially (i.e. like 1e-2,1e-3,1e-4,...)
	atols = exp10.(log10(max_atol):-1.0:log10(min_possible_atol))
	rtols = exp10.(log10(max_rtol):-1.0:log10(min_possible_rtol))

	min_achieved_atol, min_achieved_rtol = findminimum_precision(s1, s2, atols, rtols)

	# Search the order of magnitude linearly to get a more precise estimate
	atols = LinRange(min_achieved_atol, 0.1min_achieved_atol, 10)
	rtols = LinRange(min_achieved_rtol, 0.1min_achieved_rtol, 10)

	return findminimum_precision(s1, s2, atols, rtols)
end

function Base.show(io::IO, ::MIME"text/plain", t::ConvergenceTest)
	println(io, "Convergence Test")
	methodstring = repr("text/plain", t.method)
	maxtime = round(Int64, t.maxtime)
	str = """
	$(getshortname(t.start))
	method: $(methodstring)
	atolgoal: $(t.atolgoal)
	rtolgoal: $(t.rtolgoal)
	maxtime: $maxtime
	maxiter: $(t.maxiterations)
	testdatafile: $(t.testdatafile)
	"""
	print(io, prepend_spaces(str, 2))
end

function Base.show(io::IO, ::MIME"text/plain", r::ConvergenceTestResult)

	println(io, "Convergence Test Result:")

	startparams = "None"
	endparams = "None"
	if !isempty(r.test.completedsims)
		startparams = r.test.completedsims[1] |> printparamsSI
		endparams = r.test.completedsims[end] |> printparamsSI
	end

	str = """
	return code: $(r.retcode)
	achieved tolerances:
	  atol: $(r.min_achieved_atol) 
	  rtol: $(r.min_achieved_rtol)
	elapsed time: $(r.elapsed_time_sec)s
	number of simulations: $(r.iterations)

	"""

	print(io, prepend_spaces(str, 1))
	Base.show(io, "text/plain", r.test)
end

function Base.show(io::IO, ::MIME"text/plain", m::LinearTest)
	println(io, "Linear convergence test method (+$(m.shift)):")
	str = print_iterated_symbolsequence(x -> nextvalue(x, m), m.parameter)
	print(io, str)
end

function Base.show(io::IO, ::MIME"text/plain", m::PowerLawTest)
	println(io, "Power-law convergence test method (*$(m.multiplier)):")
	str = print_iterated_symbolsequence(x -> nextvalue(x, m), m.parameter)
	print(io, " - " * str)
end

function print_iterated_symbolsequence(f::Function, s::Symbol; sigdigits = 3, n = 5)
	seq = [1.0]
	for i in 1:n
		push!(seq, f(seq[end]))
	end
	str = join(["$(round(v,sigdigits=sigdigits))$s, " for v in seq])
	return "[" * str * "...]"
end
