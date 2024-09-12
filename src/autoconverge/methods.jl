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
	resume = false)

	g 			= file["completedsims"]
	done_sims 	= [load_obj_hdf5(g[s]) for s in keys(g)]
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
		path=path,
		altpath = altpath,
		completedsims = resume ? done_sims : empty([start]))
end

function dryrun(file::String,outpath::Union{Nothing,String}=nothing;kwargs...)
	h5open(file,"r") do f
		return dryrun(f,outpath;kwargs...)
	end
end

function dryrun(
	file::Union{HDF5.File, HDF5.Group},
	outpath::Union{Nothing, String}=nothing;
	atolgoal::Real = read(file,"atolgoal"),
	rtolgoal::Real = read(file,"rtolgoal"))

	g 			= file["completedsims"]
	done_sims 	= [load_obj_hdf5(g[s]) for s in keys(g)]

	sort!(done_sims,by=getsimindex)


	t = ConvergenceTest(done_sims[1],LinearChunked();
					method = load_obj_hdf5(file["method"]),
					rtolgoal = rtolgoal,
					atolgoal = atolgoal,
					maxtime = read(file,"maxtime"),
					maxiterations = read(file,"maxiterations"),
					path = outpath)
	push!(t.completedsims,done_sims[1])

	h5open(t.testdatafile,"cw") do f
		ensuregroup(f,"completedsims")
		savedata_hdf5(t.method,f)
		f["atolgoal"] 		= t.atolgoal
		f["rtolgoal"] 		= t.rtolgoal
		f["maxtime"]  		= t.maxtime
		f["maxiterations"] 	= t.maxiterations
		f["testdatafile"] 	= t.testdatafile
		savedata_hdf5(t.start,create_group(f,"start"))
	end
	

	!isnothing(outpath) && Damysos.savedata(t,done_sims[1])
	
	for (s,i) in zip(done_sims[2:end],1:length(done_sims)-1)
		achieved_tol = findminimum_precision(t)
		if converged(t)
			@info "Converged at iteration $i"
			achieved_tol = findminimum_precision(t)
			r = ConvergenceTestResult(
				t,
				ReturnCode.success,
				achieved_tol...,
				0.0,
				i,
				s.numericalparams)
			!isnothing(outpath) &&  Damysos.savedata(r)
			return r
		else
			push!(t.completedsims, s)
			!isnothing(outpath) &&  Damysos.savedata(t,s)
			@info """ 
			- Iteration $i of maximum of $(t.maxiterations)
			- Current value: $(currentvalue(t.method,s)) $(getname(t.method)))
			- Current atol: $(achieved_tol[1])
			- Current rtol: $(achieved_tol[2])
			"""
		end
	end
	@warn "No convergence achieved"
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
		t.completedsims[end].numericalparams)
	!isnothing(outpath) &&  Damysos.savedata(r)
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
	return any(retcode .== [
		ReturnCode.success,
		ReturnCode.maxtime,
		ReturnCode.maxiter,
		ReturnCode.failed,
		ReturnCode.exception])
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

	producer = let t=test,sd=savedata,sp=saveplots
		c::Channel -> _run!(c, t, t.method; savedata = sd, saveplots = sp)
	end 

	prod_taskref			= Ref{Task}();
	results 	 			= Channel{Simulation}(producer;taskref = prod_taskref)
	pollint 	 			= minimum((60.0, test.maxtime / 10))
	elapsed_time 			= 0.0

	while elapsed_time < test.maxtime && isopen(results)

		remaining_time = test.maxtime - elapsed_time

		elapsed_time += @elapsed timedwait(
			() -> isready(results) || !isopen(results), remaining_time, pollint = pollint)

		if isready(results)
			elapsed_time += @elapsed for sim in results
				push!(test.completedsims,sim)
				savedata && Damysos.savedata(test, sim)
				saveplots && Damysos.savedata(sim)
			end
		end
	end

	close(results)

	istaskfailed(prod_taskref[]) && Base.show_task_exception(stderr,prod_taskref[])

	return postrun!(test, elapsed_time, istaskfailed(prod_taskref[]); savedata = savedata)
end

function _run!(
	c::Channel,
	test::ConvergenceTest,
	method::Union{PowerLawTest, LinearTest};
	savedata = true,
	saveplots = false)

	done_sims        = test.completedsims
	currentiteration = length(done_sims)

	while currentiteration < test.maxiterations && isopen(c)

		currentiteration += 1
		currentsim = currentiteration == 1 ? test.start : next(done_sims[end], method)

		@debug "Check index: $(currentiteration) ?= $(getsimindex(currentsim))"

		run!(currentsim,test.allfunctions[currentiteration],test.solver;
			showinfo=false,
			savedata=false,
			saveplots=saveplots)

		put!(c, currentsim)		
		push!(done_sims, currentsim)
		achieved_tol = findminimum_precision(test)
		@info """ 
			- Iteration $currentiteration of maximum of $(test.maxiterations)
			- Current value: $(currentvalue(method,currentsim)) $(getname(method)))
			- Current atol: $(achieved_tol[1])
			- Current rtol: $(achieved_tol[2])
		"""
		converged(test) && break
	end
	close(c)
	return nothing
end

function postrun!(test::ConvergenceTest, elapsedtime_seconds::Real, exception_thrown::Bool;
	savedata = true)

	if converged(test)
		@info """
		## Converged after $(round(elapsedtime_seconds/60,sigdigits=3))min and \
		$(length(test.completedsims)) iterations"""
		retcode = ReturnCode.success
	elseif exception_thrown
		@warn "An exception was thrown during the ConvergenceTest."
		retcode = ReturnCode.exception
	elseif elapsedtime_seconds > test.maxtime
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

function converged(s1::Simulation,s2::Simulation;atol=DEFAULT_ATOL,rtol=DEFAULT_RTOL)
	return isapprox(s1,s2;atol = atol,rtol = rtol)
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
			test.completedsims[end];
			min_atol = test.atolgoal,
			min_rtol = test.rtolgoal)
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
	min_atol = 1e-12,
	min_rtol = 1e-12)

	p1 = getparams(s1)
	p2 = getparams(s2)

	min_possible_atol = maximum([p1.atol, p2.atol, min_atol])
	min_possible_rtol = maximum([p1.rtol, p2.rtol, min_rtol])

	# Sweep the range of tolerance exponentially (i.e. like 1e-2,1e-3,1e-4,...)
	atols = exp10.(log10(max_atol):-1.0:log10(min_possible_atol))
	rtols = exp10.(log10(max_rtol):-1.0:log10(min_possible_rtol))

	min_achieved_atol, min_achieved_rtol = findminimum_precision(s1, s2, atols, rtols)

	# Search the order of magnitude linearly to get a more precise estimate
	atols = LinRange(min_achieved_atol, 0.1min_achieved_atol, 100)
	rtols = LinRange(min_achieved_rtol, 0.1min_achieved_rtol, 100)

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
	seq = String["kxmax, "]
	for i in 1:5
		push!(seq,"kxmax + $(round(i*m.shift,sigdigits=3)), ")
	end
	print(io, " - [" * join(seq) * "...]")
end

function Base.show(io::IO, ::MIME"text/plain", m::PowerLawTest)
	println(io, "Power-law convergence test method (*$(m.multiplier)):")
	str = join(["$(round(m.multiplier^n,sigdigits=3))$(m.parameter), " for n in 0:5])
	print(io, " - [" * str * "...]")
end

