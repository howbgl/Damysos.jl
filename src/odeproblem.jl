
export buildensemble_linear
export buildsimvector_linear

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

