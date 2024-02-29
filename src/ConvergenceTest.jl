export ConvergenceTest
export ConvergenceTestMethod
export SequentialTest

abstract type ConvergenceTestMethod end

struct SequentialTest <: ConvergenceTestMethod
    parametersequence::Vector{Symbol}
end

struct ConvergenceTest{T<:Real}
    simlist::Vector{Simulation{T}}
    method::ConvergenceTestMethod
    atolgoal::T
    rtolgoal::T
    datapath::String
    plotpath::String
    id::String
end

function ConvergenceTest(
    sims::Vector{Simulation{T}},
    atolgoal::Real,
    rtolgoal::Real,
    method::ConvergenceTestMethod,
    dpath::String,
    ppath::String,
    id::String)
    
    ConvergenceTest(sims,convert(T,atolgoal),convert(T,rtolgoal),method,dpath,ppath,id)
end

function ConvergenceTest(
    sims::Vector{Simulation{T}},
    method=SequentialTest([:dt]),
    atolgoal::Real=1e-12,
    rtolgoal::Real=1e-8)
    
    isempty(sims) && throw(ArgumentError("Simulation vector is empty."))

    dpath = droplast(joinpath(sims[1].datapath,"convergencetests"))
    ppath = droplast(joinpath(sims[1].plotpath,"convergencetests"))
    id    = "dt_"*sims[1].id

    ConvergenceTest(sims,method,atolgoal,rtolgoal,dpath,ppath,id)
end

function ConvergenceTest(
    sims::Simulation{T},
    method=SequentialTest([:dt]),
    atolgoal::Real=1e-12,
    rtolgoal::Real=1e-8)

    ConvergenceTest([sims],method,atolgoal,rtolgoal)    
end