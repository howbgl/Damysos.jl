
export LinearChunked

"""
    CPULinearChunked{T}

Represents an integration strategy for k-space via simple midpoint sum.

# Fields
- `kchunksize::T`: number of k-points in one chunk. Every task/worker gets one chunk. 
- `algorithm::SciMLBase.BasicEnsembleAlgorithm`: algorithm for the [`EnsembleProblem`](@ref)

# Examples
```jldoctest
julia> solver = CPULinearChunked(256)
CPULinearChunked{Int64}(256)
```

# See also
[`EnsembleProblem`](@ref), [`EnsembleThreads`](@ref), [`EnsembleDistributed`](@ref)
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

    prerun!(sim)

    @info "Using CPUEnsembleChunked with k-chunks of size $(solver.kchunksize)"

    prob,kchunks = buildensemble_chunked_linear(sim,functions...;
        kchunk_size=solver.kchunksize)
    
    res = solve(
        prob,
        nothing,
        choose_threaded_or_distributed();
        trajectories = length(kchunks),
        saveat = gettsamples(sim.numericalparams),
        abstol = sim.numericalparams.atol,
        reltol = sim.numericalparams.rtol)

    write_ensemblesols_to_observables!(sim,res.u)

    postrun!(sim;savedata=savedata,saveplots=saveplots)

    return sim.observables 
end

function define_functions(sim::Simulation,::LinearChunked)

    ccex,cvex = buildrhs_cc_cv_x_expression(sim)
    return @eval [
        (cc,cv,kx,ky,t) -> $ccex,
        (cc,cv,kx,ky,t) -> $cvex,
        (p,t) -> $(buildbzmask_expression_upt(sim)),
        (u,p,t) -> $(buildobservable_expression_upt(sim))]
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