export ConvergenceTest
export ConvergenceTestMethod
export ConvergenceTestResult
export LinearTest
export PowerLawTest

abstract type ConvergenceTestMethod end


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
        solver::DamysosSolver=LinearChunked(),
        method::ConvergenceTestMethod=PowerLawTest(:dt,0.5),
        atolgoal::Real=1e-12,
        rtolgoal::Real=1e-8,
        maxtime::Union{Real,Unitful.Time}=600,
        maxiterations::Integer=64;
        altpath=joinpath(pwd(),start.datapath))

        maxtime = maxtime isa Real ? maxtime : ustrip(u"s",maxtime)
        
        fns = []
        s   = deepcopy(start)

        for i in 1:maxiterations
            f = define_functions(s,solver)
            s = next(s,method)
            push!(fns,f)
        end

        (success,path) = ensurepath([start.datapath,altpath])
        !success && throw(ErrorException("could not create neceesary data directory"))

        file = h5open(
            joinpath(
                path,
                "convergencetest_$(getname(start))_$(getname(method)).hdf5"),
            "cw")
        filepath = file.filename
        close(file)

        @reset start.datapath = joinpath(start.datapath,"start")
        @reset start.plotpath = joinpath(start.plotpath,"start")
        @reset start.id       = "start_$(start.id)"

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

struct LinearTest{T<:Real} <: ConvergenceTestMethod
    parameter::Symbol
    shift::T
end

struct PowerLawTest{T<:Real} <: ConvergenceTestMethod
    parameter::Symbol
    multiplier::T
end

struct ConvergenceTestResult
    test::ConvergenceTest
    success::Bool
    min_achieved_atol::Real
    min_achieved_rtol::Real
end


nextvalue(oldvalue::Real,method::PowerLawTest) = method.multiplier * oldvalue
nextvalue(oldvalue::Real,method::LinearTest)   = oldvalue + method.shift

function currentvalue(m::Union{PowerLawTest,LinearTest},sim::Simulation)
    return getproperty(sim.numericalparams,m.parameter)
end

function getfilename(m::Union{PowerLawTest,LinearTest},sim::Simulation) 
    return "$(m.parameter)=$(currentvalue(m,sim))_$(round(now(),Dates.Second))"
end


getname(t::ConvergenceTest) = "convergencetest_$(getname(t.start))_$(getname(t.method))"
getname(m::PowerLawTest)    = "PowerLawTest_$(m.parameter)"
getname(m::LinearTest)      = "LinearTest_$(m.parameter)"

function next(
    sim::Simulation,
    method::Union{PowerLawTest,LinearTest},
    parentdatapath::String=droplast(sim.datapath),
    parentplotpath::String=droplast(sim.plotpath))
    
    oldparam = getproperty(sim.numericalparams,method.parameter)
    opt      = PropertyLens(method.parameter)
    newparam = nextvalue(oldparam,method)
    params   = set(deepcopy(sim.numericalparams),opt,newparam)
    id       = "$(method.parameter)=$newparam"
    
    Simulation(
        sim.liouvillian,
        sim.drivingfield,
        params,
        zero.(sim.observables),
        sim.unitscaling,
        id,
        joinpath(parentdatapath,id),
        joinpath(parentplotpath,id))
end

function run!(
    test::ConvergenceTest;
    savetestresult=true,
    savealldata=true,
    savelastdata=true)
    
    @info "## Starting "*repr("text/plain",test)

    result = _run!(
        test,
        test.method;
        savetestresult=savetestresult,
        savesimdata=savealldata)

    if !savealldata && savelastdata
        savedata(test,test.completedsims[end])
        savemetadata(test.completedsims[end])
    end
    return result
end

function _run!(
    test::ConvergenceTest,
    method::Union{PowerLawTest,LinearTest};
    savetestresult=true,
    savesimdata=true)

    @info repr("text/plain",method)
    
    currentiteration    = 0
    elapsedtime_seconds = 0.0
    start               = test.start
    done_sims           = test.completedsims

    while currentiteration < test.maxiterations && elapsedtime_seconds < test.maxtime

        currentiteration += 1
        currentsim    = isempty(done_sims) ? start : next(done_sims[end],method)
        
        elapsedtime_seconds += @elapsed run!(
            currentsim,
            test.allfunctions[currentiteration],
            test.solver;
            saveplots=false,
            savedata=false)

        elapsed_round = round(elapsedtime_seconds/60,sigdigits=3)
        max_round     = round(test.maxtime/60,sigdigits=3)
        @info """
        - $(elapsed_round)min of $(max_round)min elapsed 
        - Iteration $currentiteration of maximum of $(test.maxiterations)
        """
        if savesimdata
            savedata(test,currentsim)
            savemetadata(currentsim)
        end
        push!(done_sims,currentsim)
        converged(test) && break
    end

    if converged(test)
        @info """
        ## Converged after $(round(elapsedtime_seconds/60,sigdigits=3))min and \
        $currentiteration iterations"""
    elseif currentiteration > maxiterations
        @warn "Maximum number of iterations ($maxiterations) reached, aborting."
    elseif elapsedtime_seconds > test.maxtime
        @warn "Maximum duration exceeded, aborting."
    else
        @warn "Something very weird happened..."
    end

    achieved_tol = length(test.completedsims) < 2 ? (Inf,Inf) : findminimum_precision(
        test.completedsims[end-1],
        test.completedsims[end])
    
    result = ConvergenceTestResult(test,converged(test),achieved_tol...)
    savetestresult && savedata(result)
    return result
end

function converged(test::ConvergenceTest)
    length(test.completedsims) < 2 ? false : isapprox(
        test.completedsims[end-1],
        test.completedsims[end];
        atol=test.atolgoal,
        rtol=test.rtolgoal)
end

function worst(results::Vector{ConvergenceTestResult},test::ConvergenceTest)
    min_achieved_atol = maximum([r.min_achieved_atol for r in results])
    min_achieved_rtol = maximum([r.min_achieved_rtol for r in results])
    success           = all([r.success for r in results])
    ConvergenceTestResult(test,success,min_achieved_atol,min_achieved_rtol)
end


function findminimum_precision(
    s1::Simulation,
    s2::Simulation,
    atols::AbstractVector{<:Real},
    rtols::AbstractVector{<:Real})

    !isapprox(s1,s2;atol=atols[1],rtol=rtols[1]) && return (Inf,Inf)

    min_achieved_atol = atols[1]
    min_achieved_rtol = rtols[1]

    # First find the lowest atol, since that is usually less problematic
    for atol in atols
        if isapprox(s1,s2;atol=atol,rtol=rtols[1])
            min_achieved_atol = atol
        else
            break
        end
    end
    for rtol in rtols
        if isapprox(s1,s2;atol=min_achieved_atol,rtol=rtol)
            min_achieved_rtol = rtol
        else
            break
        end
    end

    return (min_achieved_atol,min_achieved_rtol)
end

function findminimum_precision(s1::Simulation,s2::Simulation;max_atol=0.1,max_rtol=0.1)

    p1 = getparams(s1)
    p2 = getparams(s2)

    min_possible_atol = maximum([p1.atol,p2.atol])
    min_possible_rtol = maximum([p1.rtol,p2.rtol])

    # Sweep the range of tolerance exponentially (i.e. like 1e-2,1e-3,1e-4,...)
    atols = exp10.(log10(max_atol):-1.0:log10(min_possible_atol))
    rtols = exp10.(log10(max_rtol):-1.0:log10(min_possible_rtol))

    min_achieved_atol,min_achieved_rtol = findminimum_precision(s1,s2,atols,rtols)
    
    # Search the order of magnitude linearly to get a more precise estimate
    atols = LinRange(min_achieved_atol,0.1min_achieved_atol,10)
    rtols = LinRange(min_achieved_rtol,0.1min_achieved_rtol,10)

    return findminimum_precision(s1,s2,atols,rtols)
end

function Base.show(io::IO,::MIME"text/plain",t::ConvergenceTest)
    println(io,"Convergence Test" |> escape_underscores)
    methodstring = repr("text/plain",t.method)
    str = """
    - $(getshortname(t.start))
    - method: $(methodstring)
    - atolgoal: $(t.atolgoal)
    - rtolgoal: $(t.rtolgoal)
    - testdatafile: $(t.testdatafile)
    """ |> escape_underscores
    print(io,prepend_spaces(str,2))
end

function Base.show(io::IO,::MIME"text/plain",r::ConvergenceTestResult)
    println(io,"Convergence Test Result:" |> escape_underscores)
    startparams = "None"
    endparams = "None"
    if !isempty(r.test.completedsims)
        startparams = r.test.completedsims[1] |> printparamsSI |> escape_underscores
        endparams = r.test.completedsims[end] |> printparamsSI |> escape_underscores 
    end
    str = """
    - success: $(r.success)
    - achieved tolerances:
        * atol: $(r.min_achieved_atol) 
        * rtol: $(r.min_achieved_rtol)
    - number of simulations: $(length(r.test.completedsims))

    First simulation: 
    $(prepend_spaces(startparams,1))

    Last simulation:  
    $(prepend_spaces(endparams,1))
    """ |> escape_underscores
    print(io,prepend_spaces(str,1))
end

function Base.show(io::IO,::MIME"text/plain",m::LinearTest)
    println(io,"Linear convergence test method (+$(m.shift)):")
    str = print_iterated_symbolsequence(x -> nextvalue(x,m),m.parameter)
    print(io,str)
end

function Base.show(io::IO,::MIME"text/plain",m::PowerLawTest)
    println(io,"Power-law convergence test method (*$(m.multiplier)):")
    str = print_iterated_symbolsequence(x -> nextvalue(x,m),m.parameter)
    print(io," - " * str)
end

function print_iterated_symbolsequence(f::Function,s::Symbol;sigdigits=3,n=5)
    seq = [1.0]
    for i in 1:n
        push!(seq,f(seq[end]))
    end
    str = join(["$(round(v,sigdigits=sigdigits))$s, " for v in seq])
    return "[" * str * "...]"
end
