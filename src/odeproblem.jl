const DEFAULT_K_CHUNK_SIZE = 256

export buildensemble_chunked_linear
export buildensemble_linear
export buildsimvector_linear
export ntrajectories
export reduction

DEFAULT_REDUCTION(u, data, I) = (append!(u,sum(data)),false)


function buildensemble_chunked_linear(
    sim::Simulation,
    rhs_cc::Function,
    rhs_cv::Function,
    bzmask::Function,
    obsfunction::Function,
    kchunk_size::Integer=DEFAULT_K_CHUNK_SIZE)

    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    ks             = [getkgrid_point(i,kxs,kys) for i in 1:ntrajectories(sim)]
    kbatches       = subdivide_vector(ks,kchunk_size)
    tspan          = gettspan(sim.numericalparams)
    u0             = zeros(Complex{eltype(kxs)},2length(kbatches[1]))

    function f(du,u,p,t)
        for i in 1:length(p)
            du[2i-1]    = rhs_cc(u[2i-1],u[2i],p[i][1],p[i][2],t)
            du[2i]      = rhs_cv(u[2i-1],u[2i],p[i][1],p[i][2],t)
        end
    end

    function observables_out(sol,i)

        p       = sol.prob.p
        weigths = zeros(eltype(p[1]),length(p))
        obs     = []

        for (i,u,t) in zip(1:length(sol.u),sol.u,sol.t)

            weigths = bzmask.(p,t)
            rho     = reinterpret(SVector{2,eltype(u)},reshape(u,(2,:)))' .* weigths
            push!(obs,sum(obsfunction.(rho,p,t)))
        end

        return (obs,false)
    end

    prob            = ODEProblem{true}(f,u0,tspan,kbatches[1])
    ensprob         = EnsembleProblem(
        prob,
        prob_func   = let kb=kbatches
            (prob,i,repeat) -> remake(
                prob,
                p = kb[i],
                u0 = zeros(Complex{eltype(kb[i][1])},2length(kb[i])))
        end,
        output_func = observables_out,
        reduction   = (u, data, I) -> (append!(u,sum(data)),false),
        safetycopy  = false)
    
    return ensprob,kbatches
end



function buildensemble_linear(
    sim::Simulation,
    rhs::Function,
    bzmask::Function,
    obsfunction::Function,
    reduction::Function=DEFAULT_REDUCTION)

    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    tspan          = gettspan(sim.numericalparams)
    u0             = SA[zero(Complex{eltype(kxs)}),zero(Complex{eltype(kxs)})]
    prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))

    ensprob = EnsembleProblem(
        prob,
        prob_func   = (prob,i,repeat) -> remake(prob,p = getkgrid_point(i,kxs,kys)),
        output_func = (sol,i) -> begin
            ([
                bzmask.(sol.prob.p[1],sol.prob.p[2],sol.t) .*
                obsfunction.(sol.u,sol.prob.p[1],sol.prob.p[2],sol.t)],
            false) 
        end,
        reduction = reduction,
        u_init      = [],
        safetycopy  = false)

    return ensprob
end

function buildensemble_plain_linear(sim::Simulation,rhs::Function)

    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    tspan          = gettspan(sim.numericalparams)
    u0             = SA[zero(Complex{eltype(kxs)}),zero(Complex{eltype(kxs)})]
    prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))

    ensprob = EnsembleProblem(
        prob,
        prob_func   = (prob,i,repeat) -> remake(prob,p = getkgrid_point(i,kxs,kys)),
        output_func = (sol,i) -> (sol.u, false),
        u_init      = [],
        safetycopy  = false)

    return ensprob
end

function buildsimvector_linear(
    sim::Simulation,
    rhs::Function)

    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    tspan          = gettspan(sim.numericalparams)
    u0             = SA[zero(Complex{eltype(kxs)}),zero(Complex{eltype(kxs)})]
    prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))

    return [remake(prob,p=getkgrid_point(i,kxs,kys)) for i in 1:ntrajectories(sim)]
end

function runparallel_pairwise(
    sim::Simulation,
    rhs::Function,
    bzmask::Function,
    obsfunction::Function;
    kwargs...)

    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    ks             = [getkgrid_point(i,kxs,kys) for i in 1:ntrajectories(sim)]
    ts             = gettsamples(sim.numericalparams)
    tspan          = gettspan(sim.numericalparams)
    u0             = SA[zero(Complex{eltype(kxs)}),zero(Complex{eltype(kxs)})]
    prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))
        
    runparallel_pairwise(prob,ks,ts,bzmask,obsfunction;kwargs...)
end

function runparallel_pairwise(
    prob,
    ks::Vector{<:SVector{2,<:Real}},
    ts::AbstractVector{<:Real},
    bzmask::Function,
    obsfunction::Function;
    kwargs...)
    
    n = length(ks)
    if n <= 128
        res = []
        for k in ks
            u = solve(remake(prob,p=k),saveat=ts).u
            u .*= bzmask.(k[1],k[2],ts)
            push!(res,obsfunction.(u,k[1],k[2],ts))
        end
        return sum(res)
    else @sync begin
        m::Int64 = floor(n/2)
        s1 = Dagger.@spawn runparallel_pairwise(
            deepcopy(prob),
            deepcopy(ks[1:m]),
            deepcopy(ts),
            bzmask,
            obsfunction;
            kwargs...)
        s2 = Dagger.@spawn runparallel_pairwise(
            deepcopy(prob),
            deepcopy(ks[m+1:end]),
            deepcopy(ts),
            bzmask,
            obsfunction;
            kwargs...)
        end
        return Dagger.@spawn s1 + s2
    end
end

function parallelsolve(problems::Vector{<:ODEProblem};kwargs...)
    
    sols = []
    @sync for p in problems
        sol = Dagger.@spawn solve(p;kwargs...)
        push!(sols,sol)
    end
    return fetch.(sols)
end

ntrajectories(sim) = getnkx(sim.numericalparams) * getnky(sim.numericalparams)