export ConvergenceTest
export ConvergenceTestMethod
export ConvergenceTestResult
export LinearTest
export PowerLawTest
export run!
export SequentialTest

abstract type ConvergenceTestMethod end


struct ConvergenceTest{T<:Real}
    start::Simulation{T}
    method::ConvergenceTestMethod
    atolgoal::T
    rtolgoal::T
    datapath::String
    plotpath::String
    id::String
    completedsims::Vector{Simulation{T}}
    parameterhistory::DataFrame
end


struct SequentialTest <: ConvergenceTestMethod
    testsequence::Vector{ConvergenceTestMethod}
end

struct LinearTest{T<:Real} <: ConvergenceTestMethod
    parameter::Symbol
    shift::T
end

struct PowerLawTest{T<:Real} <: ConvergenceTestMethod
    parameter::Symbol
    multiplier::T
end

struct ConvergenceTestResult{T<:Real}
    test::ConvergenceTest{T}
    success::Bool
    min_achieved_atol::T
    min_achieved_rtol::T
end


function ConvergenceTest(
    start::Simulation{T},
    method::ConvergenceTestMethod,
    atolgoal::Real,
    rtolgoal::Real,
    dpath::String,
    ppath::String,
    id::String) where {T<:Real}
    
    atolg = convert(T,atolgoal)
    rtolg = convert(T,rtolgoal)
    ConvergenceTest(start,method,atolg,rtolg,dpath,ppath,id,empty([start]),DataFrame())
end

function ConvergenceTest(
    start::Simulation{T},
    method::ConvergenceTestMethod=PowerLawTest{T}(:dt,0.5),
    atolgoal::Real=1e-12,
    rtolgoal::Real=1e-8) where {T<:Real}

    dpath = joinpath(start.datapath,"convergencetests")
    ppath = joinpath(start.plotpath,"convergencetests")
    id    = "dt_"*start.id

    ConvergenceTest(start,method,atolgoal,rtolgoal,dpath,ppath,id)
end

function matchpaths(test::ConvergenceTest)
    start = test.start
    if start.datapath != test.datapath
        @reset start.datapath = joinpath(test.datapath,"start")
    end
    if start.plotpath != test.plotpath
        @reset start.plotpath = joinpath(test.plotpath,"start")
    end
    args = (test.method,test.atolgoal,test.rtolgoal,test.datapath,test.plotpath,test.id)
    return ConvergenceTest(start,args...)
end

nextvalue(oldvalue::Real,method::PowerLawTest) = method.multiplier * oldvalue
nextvalue(oldvalue::Real,method::LinearTest)   = oldvalue + method.shift

function next(sim::Simulation,method::Union{PowerLawTest,LinearTest})
    
    oldparam = getproperty(sim.numericalparams,method.parameter)
    opt      = PropertyLens(method.parameter)
    newparam = nextvalue(oldparam,method)
    params   = set(deepcopy(sim.numericalparams),opt,newparam)
    id       = "$(method.parameter)=$newparam"
    dpath    = joinpath(droplast(sim.datapath),id)
    ppath    = joinpath(droplast(sim.plotpath),id)
    
    Simulation(
        sim.hamiltonian,
        sim.drivingfield,
        params,
        zero.(sim.observables),
        sim.unitscaling,
        sim.dimensions,
        id,
        dpath,
        ppath)
end

function run!(test::ConvergenceTest,maxiterations::Integer=10,maxtime::Real=30*60;
    sequentialsim=true,
    savealldata=true,
    savelastdata=true)
    
    @info "## Starting "*repr("text/plain",test)

    result = run!(
        matchpaths(test),
        test.method,
        maxiterations,
        maxtime;
        sequentialsim=sequentialsim,
        savesimdata=savealldata)
    
    addhistory!(test)

    if !savealldata && savelastdata
        savedata(result.test.completedsims[end])
    end
    return result
end

function run!(
    test::ConvergenceTest,
    method::Union{PowerLawTest,LinearTest},
    maxiterations::Integer,
    maxduration::Real;
    savetestresult=true,
    savesimdata=true,
    sequentialsim=true)

    @info repr("text/plain",method)
    
    currentiteration    = 0
    elapsedtime_seconds = 0.0

    while currentiteration < maxiterations && elapsedtime_seconds < maxduration

        elapsed_round       = round(elapsedtime_seconds/60,sigdigits=3)
        max_round           = round(maxduration/60,sigdigits=3)

        if isempty(test.completedsims)
            push!(test.completedsims,test.start)
        else
            push!(test.completedsims,next(test.completedsims[end],method))
        end
        
        elapsedtime_seconds += @elapsed run_simulation!(
            test.completedsims[end];
            saveplots=false,
            savedata=savesimdata,
            sequential=sequentialsim)
        currentiteration    += 1
        @info """
        - $(elapsed_round)min of $(max_round)min elapsed 
        - Iteration $currentiteration of maximum of $maxiterations
        """
        converged(test) && break
    end

    if converged(test)
        @info """
        ## Converged after $(round(elapsedtime_seconds/60,sigdigits=3))min and\
        $currentiteration iterations"""
    elseif currentiteration > maxiterations
        @warn "Maximum number of iterations ($maxiterations) reached, aborting."
    elseif elapsedtime_seconds > maxduration
        @warn "Maximum duration exceeded, aborting."
    else
        @warn "Something very weird happened..."
    end

    achieved_tol = length(test.completedsims) < 2 ? (Inf,Inf) : findminimum_precision(
        test.completedsims[end-1],
        test.completedsims[end])
    
    result = ConvergenceTestResult(test,converged(test),achieved_tol...)
    if savetestresult
        savedata(result,joinpath(test.datapath,"$(method.parameter)-testresult.txt"))
    end
    return result
end

function run!(
    test::ConvergenceTest,
    method::SequentialTest,
    maxiterations::Integer,
    maxduration::Real;
    savetestresult=true,
    savesimdata=true,
    sequentialsim=true)

    currentiteration    = 0
    elapsedtime_seconds = 0.0
    results             = ConvergenceTestResult[]

    for m in method.testsequence
        @info """
        Starting sequence of $(length(method.testsequence)) convergence tests
        """

        remainingiterations     = maxiterations - currentiteration
        remainingduration       = maxduration - elapsedtime_seconds
        time = @elapsed result  = run!(
            test,
            m,
            remainingiterations,
            remainingduration;
            savesimdata=savesimdata,
            sequentialsim=sequentialsim)
        elapsedtime_seconds     += time
        currentiteration        = length(result.test.completedsims)

        push!(results,result)
        !result.success && break
        if currentiteration > maxiterations || elapsedtime_seconds > maxduration
            break
        end
    end
    result = worst(results,test)

    if savetestresult
        savedata(result,joinpath(test.datapath,"combined-testresult.txt"))
    end
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

function addhistory!(test::ConvergenceTest)
    
    pars = []
    for (s1,s2) in zip(test.completedsims[1:end-1],test.completedsims[2:end])
        push!(pars,diffparams(s1.numericalparams,s2.numericalparams))
    end
    changed_parameters = union(pars...)
    return DataFrame(Dict(
        [s => getproperty.(test.completedsims,s) for s in changed_parameters]))
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
    println(io,"Convergence Test ($(t.id)):" |> escape_underscores)
    methodstring = repr("text/plain",t.method)
    str = """
    - $(getshortname(t.start))
    - method: $(methodstring)
    - atolgoal: $(t.atolgoal)
    - rtolgoal: $(t.rtolgoal)
    - datapath: $(t.datapath)
    - plotpath: $(t.plotpath)
    """ |> escape_underscores
    print(io,prepend_spaces(str,2))
end

function Base.show(io::IO,::MIME"text/plain",r::ConvergenceTestResult)
    println(io,"Convergence Test Result ($(r.test.id)):" |> escape_underscores)
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

    Parameter history:
    $(r.test.parameterhistory)

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

function Base.show(io::IO,::MIME"text/plain",m::SequentialTest)
    println(io,"Sequence of convergence tests:")
    strs = string.([prepend_spaces(repr("text/plain",s),1) for s in m.testsequence])
    print(io,join(strs,"\n"))
end
