DEFAULT_K_CHUNK_SIZE = 4096

export buildensemble_linear
export buildsimvector_linear
export ntrajectories
export reduction

DEFAULT_REDUCTION(u, data, I) = (append!(u, data), false)

# function reduction(u,data,I)
#     for s in data
#         s[2] .*= s[1]
#     end
#     return (append!(u,sum(x -> getindex(x,2),data)),false)
# end


function buildensemble_linear(
    sim::Simulation,
    rhs::Function,
    bzmask::Function,
    obsfunction::Function;
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
    else
        m::Int64 = floor(n/2)
        s1 = Dagger.@spawn runparallel_pairwise(
            deepcopy(prob),
            ks[1:m],
            deepcopy(ts),
            bzmask,
            obsfunction;
            kwargs...)
        s2 = Dagger.@spawn runparallel_pairwise(
            deepcopy(prob),
            ks[m+1:end],
            deepcopy(ts),
            bzmask,
            obsfunction;
            kwargs...)
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