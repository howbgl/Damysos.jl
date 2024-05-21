
export LinearChunked
export ntrajectories

"""
    LinearChunked{T}

Represents an integration strategy for k-space via simple midpoint sum.

# Fields
- `kchunksize::T`: number of k-points in one chunk. Every task/worker gets one chunk. 
- `algorithm::SciMLBase.BasicEnsembleAlgorithm`: algorithm for the `EnsembleProblem`

# Examples
```jldoctest
julia> solver = LinearChunked(256,EnsembleThreads())
LinearChunked:
  - kchunksize: 256
  - algorithm: EnsembleThreads()
  
```

# See also
[`LinearChunked`](@ref LinearChunked), [`SingleMode`](@ref SingleMode)
"""
struct LinearChunked{T<:Integer} <: DamysosSolver 
    kchunksize::T
    algorithm::SciMLBase.BasicEnsembleAlgorithm
end
LinearChunked() = LinearChunked(DEFAULT_K_CHUNK_SIZE)
function LinearChunked(kchunksize::Integer) 
    LinearChunked(kchunksize,choose_threaded_or_distributed())
end


function run!(
    sim::Simulation,
    functions,
    solver::LinearChunked;
    savedata=true,
    saveplots=true)

    iszero(sim.dimensions) && throw(ArgumentError(
        "LinearChunked solver is only supported for 1d and 2d simulations"
    ))

    prerun!(sim;savedata=savedata,saveplots=saveplots)

    @info """
        Solver: $(repr(solver))
    """

    rhscc,rhscv = functions[1]
    fns = (rhscc,rhscv,functions[2:end]...)
    @show fns

    prob,kchunks = buildensemble(sim,solver,fns...)
    
    res = solve(
        prob,
        nothing,
        solver.algorithm;
        trajectories = length(kchunks),
        saveat = gettsamples(sim.numericalparams),
        abstol = sim.numericalparams.atol,
        reltol = sim.numericalparams.rtol,
        progress = true)

    write_ensemblesols_to_observables!(sim,res.u)

    postrun!(sim;savedata=savedata,saveplots=saveplots)

    return sim.observables 
end

function define_rhs_x(sim::Simulation,::LinearChunked)
    
    ccex,cvex = buildrhs_cc_cv_x_expression(sim)
    return (@eval (cc,cv,kx,ky,t) -> $ccex, @eval (cc,cv,kx,ky,t) -> $cvex)
end

define_bzmask(sim::Simulation,::LinearChunked) = define_bzmask(sim)

function define_observable_functions(sim::Simulation,::LinearChunked)
    expressions = [buildobservable_expression_svec_upt(sim,o) for o in sim.observables]
    return @eval (u,p,t) -> SA[$(expressions...)]
end

function observables_out(sol,bzmask,obsfunction)

    p       = sol.prob.p
    weigths = zeros(eltype(p[1]),length(p))
    obs     = []

    for (i,u,t) in zip(1:length(sol.u),sol.u,sol.t)

        weigths .= bzmask.(p,t)

        # this reinterpret code somehow produces wrong results, maybe post to 
        # stackexchange
        # rho     = reinterpret(SVector{2,eltype(u)},reshape(u,(2,:)))'

        rho     = [SA[cc,cv] for (cc,cv) in zip(u[1:2:end],u[2:2:end])]
        push!(obs,sum(weigths .* obsfunction.(rho,p,t)))
    end

    return (obs,false)
end


function buildensemble(
    sim::Simulation,
    solver::LinearChunked,
    rhs_cc::Function,
    rhs_cv::Function,
    bzmask::Function,
    obsfunction::Function)

    kbatches        = buildkgrid_chunks(sim,solver.kchunksize)
    prob            = buildode(sim,solver,kbatches[1],rhs_cc,rhs_cv)
    ensprob         = EnsembleProblem(
        prob,
        prob_func   = let kb=kbatches
            (prob,i,repeat) -> remake(
                prob,
                p = kb[i],
                u0 = zeros(Complex{eltype(kb[i][1])},2length(kb[i])))
        end,
        output_func = (sol,i) -> observables_out(sol,bzmask,obsfunction),
        reduction   = (u, data, I) -> (append!(u,sum(data)),false),
        safetycopy  = false)
    
    return ensprob,kbatches
end

function buildode(
    sim::Simulation{T},
    ::LinearChunked,
    kbatch::Vector{SVector{2,T}},
    rhs_cc::Function,
    rhs_cv::Function) where {T<:Real}

    tspan   = gettspan(sim.numericalparams)
    u0      = zeros(Complex{T},2length(kbatch))

    function f(du,u,p,t)
        for i in 1:length(p)
            @inbounds du[2i-1]    = rhs_cc(u[2i-1],u[2i],p[i][1],p[i][2],t)
            @inbounds du[2i]      = rhs_cv(u[2i-1],u[2i],p[i][1],p[i][2],t)
        end
    end

    return ODEProblem{true}(f,u0,tspan,kbatch)
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

getserialsolver(solver::LinearChunked) = LinearChunked(solver.kchunksize,EnsembleSerial())

function Base.show(io::IO,::MIME"text/plain",s::LinearChunked)
    println(io,"LinearChunked:" |> escape_underscores)
    str = """
    - kchunksize: $(s.kchunksize)
    - algorithm: $(s.algorithm)
    """ |> escape_underscores
    print(io,prepend_spaces(str,2))
end

ntrajectories(sim) = getnkx(sim.numericalparams) * getnky(sim.numericalparams)