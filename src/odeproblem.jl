DEFAULT_K_CHUNK_SIZE = 4096

export buildensemble_linear
export ntrajectories
export reduction

function reduction(u,data,I)
    for s in data
        s[4] .*= s[3]
    end
    return (append!(u,sum(x -> getindex(x,4),data)),false)
end

function buildensemble_linear(
    sim::Simulation,
    rhs::Function,
    bzmask::Function,
    obsfunction::Function,
    reduc::Function)

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
                sol.u,
                sol.t,
                bzmask.(sol.prob.p[1],sol.prob.p[2],sol.t),
                obsfunction.(sol.u,sol.prob.p[1],sol.prob.p[2],sol.t)],
            false) 
        end,
        reduction = reduc,
        u_init      = [],
        safetycopy  = false)

    return ensprob
end

ntrajectories(sim) = getnkx(sim.numericalparams) * getnky(sim.numericalparams)