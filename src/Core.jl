

export CPUEnsembleChunked
export define_functions
export run!

struct CPUEnsembleChunked{T<:Integer} <: DamysosSolver 
    kchunksize::T
end
CPUEnsembleChunked() = CPUEnsembleChunked(DEFAULT_K_CHUNK_SIZE)

include("strategies/SolveAndIntegrate.jl")

function run!(
    sim::Simulation,
    functions;
    savedata=true,
    saveplots=true)
    
    run!(
        sim,
        functions,
        CPUEnsembleChunked(),
        savedata=savedata,
        saveplots=saveplots)
end

function run!(
    sim::Simulation,
    functions,
    solver::CPUEnsembleChunked;
    savedata=true,
    saveplots=true)

    prerun!(sim)

    @info """
        ### Using CPUEnsembleChunked with $(solver.kchunksize) k-chunks"""

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

define_functions(sim::Simulation) = define_functions(sim,CPUEnsembleChunked())
function define_functions(sim::Simulation,::CPUEnsembleChunked)

    ccex,cvex = buildrhs_cc_cv_x_expression(sim)
    return @eval [
        (cc,cv,kx,ky,t) -> $ccex,
        (cc,cv,kx,ky,t) -> $cvex,
        (p,t) -> $(buildbzmask_expression_upt(sim)),
        (u,p,t) -> $(buildobservable_expression_upt(sim))]
end

function prerun!(sim::Simulation)

    @info """
        ## $(getshortname(sim)) (id: $(sim.id))

        Starting on **$(gethostname())** at **$(now())**:
        
        * threads: $(Threads.nthreads())
        * processes: $(Distributed.nprocs())
        * plotpath: $(sim.plotpath)
        * datapath: $(sim.datapath)

        $(markdown_paramsSI(sim))
        """

    checkbzbounds(sim)
    ensurepath(sim.datapath)
    ensurepath(sim.plotpath)

    resize_obs!(sim)
    zero.(sim.observables)
end

function postrun!(sim::Simulation;savedata=true,saveplots=true)
    
    normalize!.(sim.observables,(2π)^sim.dimensions)

    if savedata
        Damysos.savedata(sim)
        savemetadata(sim)
    end
    if saveplots
        plotdata(sim)
    end
end

function choose_threaded_or_distributed()
    if Threads.nthreads() > 1 && nworkers() == 1
        return EnsembleThreads()
    elseif Threads.nthreads() == 1 && nworkers() > 1
        return EnsembleDistributed()
    else
        @warn """"
        Multiple threads and processes detected. This might result in unexpected behavior.
        Using EnsembleDistributed() nonetheless."""
        return EnsembleDistributed()
    end
end