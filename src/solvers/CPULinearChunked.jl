
export CPULinearChunked

"""
    CPULinearChunked{T}

Represents an integration strategy for k-space via simple midpoint sum.

# Fields
- `kchunksize::T`: number of k-points in one chunk. Every task/worker gets one chunk. 

# Examples
```jldoctest
julia> solver = CPULinearChunked(256)
CPULinearChunked{Int64}(256)
```

"""
struct CPULinearChunked{T<:Integer} <: DamysosSolver 
    kchunksize::T
end
CPULinearChunked() = CPULinearChunked(DEFAULT_K_CHUNK_SIZE)


function run!(
    sim::Simulation,
    functions,
    solver::CPULinearChunked;
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

function define_functions(sim::Simulation,::CPULinearChunked)

    ccex,cvex = buildrhs_cc_cv_x_expression(sim)
    return @eval [
        (cc,cv,kx,ky,t) -> $ccex,
        (cc,cv,kx,ky,t) -> $cvex,
        (p,t) -> $(buildbzmask_expression_upt(sim)),
        (u,p,t) -> $(buildobservable_expression_upt(sim))]
end
