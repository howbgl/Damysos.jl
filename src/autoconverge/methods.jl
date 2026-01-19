export successful_retcode
export terminated_retcode

const DEFAULT_NAN_LIMIT 		= 128
const DEFAULT_MAX_NAN_RETRIES 	= 2

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
	maxiterations::Integer = convert(Int,read(file,"maxiterations")))

	g 			= file["completedsims"]
	done_sims 	= Simulation[load_obj_hdf5(g[s]) for s in keys(g)]
	start 		= load_obj_hdf5(file["start"])

	sort!(done_sims,by=getsimindex)

	return ConvergenceTest(
		isempty(done_sims) ? start : last(done_sims),
		solver;
		method=method,
		atolgoal=atolgoal,
		rtolgoal=rtolgoal,
		maxtime=maxtime,
		maxiterations=maxiterations,
		completedsims=done_sims)
end

function check_compatibility(sim::Simulation, m::ConvergenceTestMethod, s::DamysosSolver)
	if m isa ExtendKymaxTest 
		if s isa LinearChunked
			@warn "ExtendKymaxTest using LinearChunked is bugged atm (to be fixed soon)."
		end
		if sim.drivingfield isa Union{GaussianAPulse, GaussianEPulse} 
			if sim.drivingfield.φ == 0.0
				return nothing
			end
		end
		throw(ArgumentError(
			"""
			ExtendKymaxTest is only compatible with GaussianAPulse/GaussianEPulse along
			kx-direction (φ=0). Non-trivial BZ stuff not implemented yet.
			"""))
	else
		return nothing
	end
end

function ConvergenceTestResult(
	test::ConvergenceTest,
	retcode::ReturnCode.T,
	min_achieved_atol::Real,
	min_achieved_rtol::Real,
	elapsed_time_sec::Real,
	iterations::Integer,
	last_params::NGrid)
	
	return ConvergenceTestResult(
		test,
		retcode,
		min_achieved_atol,
		min_achieved_rtol,
		elapsed_time_sec,
		iterations,
		last_params,
		extrapolate(test).observables)
end


function dryrun(file::String, filepath::String; kwargs...)
	h5open(file,"r") do f
		return dryrun(f,filepath;kwargs...)
	end
end

function dryrun(
	file::Union{HDF5.File, HDF5.Group},
	filepath::String;
	atolgoal::Real = read(file,"atolgoal"),
	rtolgoal::Real = read(file,"rtolgoal"))

	ensuredirpath(dirname(filepath))

	g 			= file["completedsims"]
	done_sims 	= [load_obj_hdf5(g[s]) for s in keys(g)]

	sort!(done_sims,by=getsimindex)


	t = ConvergenceTest(done_sims[1],LinearChunked();
					method = load_obj_hdf5(file["method"]),
					rtolgoal = rtolgoal,
					atolgoal = atolgoal,
					maxtime = read(file,"maxtime"),
					maxiterations = Int(read(file,"maxiterations")))

	h5open(filepath,"cw") do f
		ensuregroup(f,"completedsims")
		savedata_hdf5(t.method,f)
		f["atolgoal"] 		= t.atolgoal
		f["rtolgoal"] 		= t.rtolgoal
		f["maxtime"]  		= t.maxtime
		f["maxiterations"] 	= t.maxiterations
		savedata_hdf5(t.start,create_group(f,"start"))
	end
	
	for (s,i) in zip(done_sims,1:length(done_sims))
		push!(t.completedsims, s)
		Damysos.savedata(t, s, filepath)
		achieved_tol = findminimum_precision(t)
		if converged(t)
			@info "Converged at iteration $i"
			r = ConvergenceTestResult(
				t,
				ReturnCode.success,
				achieved_tol...,
				0.0,
				i,
				s.grid)
			Damysos.savedata(r, filepath)
			return r
		else
			@info """ 
			- Iteration $i of maximum of $(t.maxiterations)
			- Current value: $(currentvalue(t.method,s)) $(getname(t.method)))
			- Current atol: $(achieved_tol[1])
			- Current rtol: $(achieved_tol[2])
			"""
		end
	end
	!converged(t) && @warn "No convergence achieved"
	achieved_tol = findminimum_precision(t)
	@info """ 
			- Achieved atol: $(achieved_tol[1])
			- Achieved rtol: $(achieved_tol[2])
			"""
	r = ConvergenceTestResult(
		t,
		ReturnCode.failed,
		achieved_tol...,
		0.0,
		length(t.completedsims),
		t.completedsims[end].grid)
	Damysos.savedata(r, filepath)
	return r
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
		if "testresult" ∈ keys(file) && "retcode" ∈ keys(file["testresult"])
			return successful_retcode(ReturnCode.T(read(file["testresult"], "retcode")))
		else
			return false
		end
	end
end

"""
    terminated_retcode(x)

Returns true if x terminated regularly.

"""
function terminated_retcode(retcode::ReturnCode.T)
	return retcode != ReturnCode.running
end

terminated_retcode(x::ConvergenceTestResult) = terminated_retcode(x.retcode)

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

parametername(m::ConvergenceTestMethod) = string(m.parameter)
parametername(m::ExtendKymaxTest) 		= parametername(m.extendmethod)

nextvalue(oldvalue::Real, method::PowerLawTest)    = method.multiplier * oldvalue
nextvalue(oldvalue::Real, method::LinearTest)      = oldvalue + method.shift
nextvalue(oldvalue::Real, method::ExtendKymaxTest) = nextvalue(oldvalue,method.extendmethod)

function getsimindex(sim::Simulation)
	m = match(r"(?<=#)\d+",sim.id)
	isnothing(m) && throw(ErrorException(
		"Could not extract simulation index from its id $(sim.id)"))
	return parse(Int,m.match)
end

currentvalue(m::ExtendKymaxTest, sim::Simulation) 		= currentvalue(m.extendmethod,sim)
currentvalue(m::ConvergenceTestMethod, sim::Simulation) = currentvalue(m, sim.grid)

function currentvalue(m::ConvergenceTestMethod, grid::NGrid)
	if m.parameter in fieldnames(typeof(grid.tgrid))
		return currentvalue(m, grid.tgrid)
	else
		return currentvalue(m, grid.kgrid)
	end
end

function currentvalue(
	m::Union{PowerLawTest, LinearTest},
	grid::KGrid)
	return getproperty(grid, m.parameter)
end

function invert_h(m::Union{PowerLawTest, LinearTest})
	if m.parameter == :kxmax || m.parameter == :kymax
		return true
	else
		return false
	end
end
invert_h(::ExtendKymaxTest) = true

function extendsim_kymax(sim::Simulation, previoussim::Simulation)
	kg 			= sim.grid.kgrid 
	extendkgrid = CartesianKGrid2dStrips(
		kg.dkx,
		kg.kxmax,
		kg.dky,
		kg.kymax,
		previoussim.grid.kgrid.kymax)
	return @set sim.grid.kgrid = extendkgrid
end

function getfilename(m::Union{PowerLawTest, LinearTest, ExtendKymaxTest}, sim::Simulation)
	return "$(m.parameter)=$(currentvalue(m,sim))_$(round(now(),Dates.Second))"
end


getname(t::ConvergenceTest) = "convergencetest_$(getname(t.start))_$(getname(t.method))"
getname(m::PowerLawTest)    = "PowerLawTest_$(m.parameter)"
getname(m::LinearTest)      = "LinearTest_$(m.parameter)"
getname(m::ExtendKymaxTest) = "ExtendKymaxTest_$(getname(m.extendmethod))"

function next(
	sim::Simulation,
	method::Union{PowerLawTest, LinearTest})

	oldparam = if method.parameter in fieldnames(typeof(sim.grid.tgrid))
		getproperty(sim.grid.tgrid, method.parameter)
	else
		getproperty(sim.grid.kgrid, method.parameter)
	end
	opt = if method.parameter in fieldnames(typeof(sim.grid.tgrid))
		opcompose(PropertyLens(:tgrid), PropertyLens(method.parameter))
	else
		opcompose(PropertyLens(:kgrid), PropertyLens(method.parameter))
	end
	newparam = nextvalue(oldparam, method)
	grid   	= set(deepcopy(sim.grid), opt, newparam)
	id       = "#$(getsimindex(sim)+1)_$(method.parameter)=$newparam"

	Simulation(
		sim.liouvillian,
		sim.drivingfield,
		grid,
		zero.(sim.observables),
		sim.unitscaling,
		id)
end
next(sim::Simulation, method::ExtendKymaxTest) = next(sim,method.extendmethod)

function prepare_hdf5_file(test::ConvergenceTest,
	filepath=joinpath(pwd(),getname(test)*"_$(test.rtolgoal).hdf5"))

	ensuredirpath(dirname(filepath))
	h5open(filepath,"cw") do file

		ensuregroup(file,"completedsims")
		for (name,object) in zip(
			["atolgoal","rtolgoal","maxtime","maxiterations"],
			[test.atolgoal,test.rtolgoal,test.maxtime,test.maxiterations])
			replace_data_hdf5(file,name,object)
		end

		if "method" ∈ keys(file)
			delete_object(file["method"])
		end
		savedata_hdf5(test.method,file)

		if "start" ∈ keys(file)
			delete_object(file["start"])
		end
		savedata_hdf5(test.start,create_group(file,"start"))
	end
end

"""
    run!(test::ConvergenceTest; kwargs...)

Run a convergence test and return the result as ConvergenceTestResult.

# Keyword Arguments
- `savedata::Bool`: save observables and simulations to disk
- `filepath::String`: path to .hdf5 file to save results
- `nan_limit::Int`: maximum tolerated number of nans in observables
- `max_nan_retries::Int`: maximum number of iterations where `nan_limit` nans are tolerated

# Returns
A ConvergenceTestResult object.

# See also
[`ConvergenceTest`](@ref)
"""
function run!(
	test::ConvergenceTest;
	savedata = true,
	filepath = joinpath(pwd(),getname(test)*"_$(test.rtolgoal).hdf5"),
	nan_limit = DEFAULT_NAN_LIMIT,
	max_nan_retries = DEFAULT_MAX_NAN_RETRIES)

	starting_time 			= now()
	
	savedata && prepare_hdf5_file(test, filepath)

	@info """
	## Starting $(repr("text/plain", test))
	$(repr("text/plain", test.method))
	"""
	printinfo(test.start,test.solver)

	producer = let t=test, nl=nan_limit
		c::Channel -> _run!(c, t, t.method; nan_limit=nl)
	end 

	prod_taskref			= Ref{Task}();
	results 	 			= Channel{Simulation}(producer;taskref = prod_taskref)
	pollint 	 			= minimum((60.0, test.maxtime / 10))
	nan_errors				= 0
	method 					= test.method
	retcode 				= ReturnCode.running
	sims 					= test.completedsims

	while !terminated_retcode(retcode) && isopen(results)

		remaining_time = test.maxtime - seconds_passed_since(starting_time)
		
		finished = timedwait(
			() -> isready(results) || !isopen(results), remaining_time, pollint = pollint)
		if finished == :timed_out
			retcode = ReturnCode.maxtime
			break
		end

		while isready(results)
			sim = take!(results)
			push!(sims,sim)

			savedata && Damysos.savedata(test, sim, filepath)

			oe = extrapolate(test)
			i  = length(sims)
			@info """ 
				- Iteration $i of maximum of $(test.maxiterations)
				- Current value: $(currentvalue(method,sim)) $(getname(method)))
				- $(repr("text/plain",oe))
			"""

			retcode = converged(test) ? ReturnCode.success : retcode
			retcode = length(sims) ≥ test.maxiterations ? ReturnCode.maxiter : retcode

			if count_nans(sim.observables) > nan_limit
				nan_errors += 1
				retcode = nan_errors > max_nan_retries ? ReturnCode.nan_abort : retcode
			end
			elapsed_time = seconds_passed_since(starting_time)
			retcode = elapsed_time + pollint > test.maxtime ? ReturnCode.maxtime : retcode
		end
	end

	close(results)

	if istaskfailed(prod_taskref[])
		Base.show_task_exception(stderr,prod_taskref[])
		retcode = ReturnCode.exception
	end

	return postrun!(test, seconds_passed_since(starting_time), retcode; 
		savedata = savedata,
		filepath = filepath)
end

function _run!(
	c::Channel,
	test::ConvergenceTest,
	method::Union{PowerLawTest, LinearTest, ExtendKymaxTest};
	nan_limit = DEFAULT_NAN_LIMIT)

	done_sims        = test.completedsims
	currentiteration = length(done_sims)
	currentsim 		 = currentiteration == 0 ? test.start : next(done_sims[end], method)
	previoussim 	 = currentiteration == 0 ? nothing : done_sims[end]

	while currentiteration < test.maxiterations && isopen(c)

		currentiteration += 1
		@debug "Check index: $(currentiteration) ?= $(getsimindex(currentsim))"
		if method isa ExtendKymaxTest && currentiteration > 1
			currentsim = extendsim_kymax(currentsim, previoussim)
		end
		
		run!(currentsim,test.allfunctions[currentiteration],test.solver;
			showinfo=false,
			savedata=false,
			saveplots=false,
			nan_limit=nan_limit)

		if method isa ExtendKymaxTest && currentiteration > 1
			currentsim.observables .= previoussim.observables .+ currentsim.observables
		end
		
		put!(c, currentsim)

		previoussim = deepcopy(currentsim)		
		currentsim 	= next(currentsim, method)
	end
	close(c)
	return nothing
end

function postrun!(test::ConvergenceTest, elapsedtime_seconds::Real, retcode;
	savedata = true,
	filepath = joinpath(pwd(),getname(test)*"_$(test.rtolgoal).hdf5"),)

	retcode = terminated_retcode(retcode) ? retcode : ReturnCode.failed
	
	oe 			 = extrapolate(test)
	achieved_tol = findminimum_precision(oe,test.atolgoal)
	
	done_sims   = test.completedsims
	last_params = isempty(done_sims) ? test.start.grid : done_sims[end].grid
	
	result = ConvergenceTestResult(
		test,
		retcode,
		achieved_tol...,
		elapsedtime_seconds,
		length(test.completedsims),
		last_params,
		oe.observables)
		
	savedata && Damysos.savedata(result, filepath)
	
	testresult_message(result)

	return result
end

function testresult_message(result::ConvergenceTestResult)

	r 		= result.retcode
	time 	= round(result.elapsed_time_sec/60,sigdigits=3)
	n 		= length(result.test.completedsims)

	if r == ReturnCode.success
		@info "## Converged after $(time)min and $n iterations."
	elseif r == ReturnCode.maxtime
		@warn "## Maximum runtime exceeded."
	elseif r == ReturnCode.maxiter
		@warn "## Maximum number of iterations ($(result.test.maxiterations)) exceeded."
	elseif r == ReturnCode.exception
		@warn "## An exception was thrown during the ConvergenceTest."
	elseif r == ReturnCode.running
		@warn "## ReturnCode.running received! Something very weird happened..."
	elseif r == ReturnCode.nan_abort
		@warn "## Too many NaNs."
	elseif r == ReturnCode.failed
		@warn "## Unkown failure (ReturnCode.failed)."
	end

	@info "$(repr("text/plain", result))"
end

function extrapolate(test::ConvergenceTest; kwargs...)
	
	res = []
	for i in 1:length(test.start.observables)
		oh_itr = [(s.observables[i],currentvalue(test.method,s)) for s in test.completedsims]
		
		push!(res,extrapolate(oh_itr, invert_h=invert_h(test.method), kwargs...))
	end
	return ObservableExtrapolation(first.(res),last.(res))
end

function converged(test::ConvergenceTest) 
	length(test.completedsims) < 2 && return false
	return converged(extrapolate(test);atol=test.atolgoal,rtol=test.rtolgoal)
end

function converged(oe::ObservableExtrapolation;rtol=DEFAULT_RTOL,atol=DEFAULT_ATOL)
	obserrs = zip(oe.observables,oe.errs)
	return all([converged(obs,errs,rtol=rtol,atol=atol) for (obs,errs) in obserrs])
end

function converged(obs::Observable,errs::Vector{<:Real};rtol=DEFAULT_RTOL,atol=DEFAULT_ATOL)
	nameserrs = zip(fieldnames(typeof(obs)),errs)
	return all([err ≤ max(atol,rtol*norm(getproperty(obs,n))) for (n,err) in nameserrs])
end

function findminimum_precision(test::ConvergenceTest)
	length(test.completedsims) < 2 && return (Inf, Inf)
	return findminimum_precision(extrapolate(test),test.atolgoal)
end

function findminimum_precision(oe::ObservableExtrapolation{T},
	atolgoal=DEFAULT_ATOL;
	max_atol=100.0,
	max_rtol=2.0) where {T<:Real}

	!converged(oe;rtol=max_rtol,atol=max_atol) && return (Inf, Inf)

	# Sweep the range of tolerance exponentially
	atols = exp10.(log10(max_atol):-0.2:log10(atolgoal))
	rtols = exp10.(log10(max_rtol):-0.1:log10(sqrt(eps(T))))

	min_achieved_atol = atols[1]
	min_achieved_rtol = rtols[1]

	# First find the lowest atol, since that is usually less problematic
	for atol in atols
		if converged(oe; atol = atol, rtol = rtols[1])
			min_achieved_atol = atol
		else
			break
		end
	end
	for rtol in rtols
		if converged(oe; atol = min_achieved_atol, rtol = rtol)
			min_achieved_rtol = rtol
		else
			break
		end
	end

	return (min_achieved_atol, min_achieved_rtol)
end

function Base.show(io::IO, ::MIME"text/plain", t::ConvergenceTest)
	println(io, "Convergence Test")
	methodstring = repr("text/plain", t.method)
	maxtime = isfinite(t.maxtime) ? round(Int64, t.maxtime) : t.maxtime
	str = """
	$(getshortname(t.start))
	method: $(methodstring)
	atolgoal: $(t.atolgoal)
	rtolgoal: $(t.rtolgoal)
	maxtime: $maxtime
	maxiter: $(t.maxiterations)
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
	seq = String["$(m.parameter), "]
	for i in 1:5
		push!(seq,"$(m.parameter) + $(round(i*m.shift,sigdigits=3)), ")
	end
	print(io, " - [" * join(seq) * "...]")
end

function Base.show(io::IO, ::MIME"text/plain", m::PowerLawTest)
	println(io, "Power-law convergence test method (*$(m.multiplier)):")
	str = join(["$(round(m.multiplier^n,sigdigits=3))$(m.parameter), " for n in 0:5])
	print(io, " - [" * str * "...]")
end

function Base.show(io::IO, ::MIME"text/plain", oe::ObservableExtrapolation)
	println(io, "ObservableExtrapolation:")
	for (o,errs) in zip(oe.observables,oe.errs) 
		println(io, " $(typeof(o)):")
		for (n,err) in zip(fieldnames(typeof(o)),errs)
			println(io, "  $n:")
			println(io, "      relerr: $(err/norm(getproperty(o,n)))")
			println(io, "      abserr: $err")
		end
	end
end
