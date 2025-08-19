export SingleMode

"""
    SingleMode

Represents the solver for a single point in k-space.

# See also
[`LinearChunked`](@ref LinearChunked), [`LinearCUDA`](@ref LinearCUDA)

"""
struct SingleMode <: DamysosSolver 
end

function solver_compatible(sim::Simulation,::SingleMode)
    return sim.dimensions == 0
end


function _run!(
    sim::Simulation,
    functions,
    solver::SingleMode)

    rhscc,rhscv = functions[1]
    k           = SA[sim.numericalparams.kx,sim.numericalparams.ky]
    prob        = buildode(sim,solver,k,rhscc,rhscv)

    
    res = solve(
        prob;
        saveat = gettsamples(sim.numericalparams),
        abstol = sim.numericalparams.atol,
        reltol = sim.numericalparams.rtol)

    for (o,f) in zip(sim.observables,functions[3])
        calculate_observable_singlemode!(sim,o,f,res)
    end

    return sim.observables 
end


function buildode(
    sim::Simulation{T},
    ::SingleMode,
    k::AbstractVector{T},
    rhs_cc::Function,
    rhs_cv::Function) where {T<:Real}

    tspan   = gettspan(sim.numericalparams)
    u0      = zeros(Complex{T},2)
    function f(du,u,p,t)
        du[1] = rhs_cc(u[1],u[2],p[1],p[2],t)
        du[2] = rhs_cv(u[1],u[2],p[1],p[2],t)
    end

    return ODEProblem{true}(f,u0,tspan,k)
end

function define_rhs_x(sim::Simulation,::SingleMode)
    
    ccex,cvex = buildrhs_cc_cv_expression(sim)
    return (@eval (cc,cv,kx,ky,t) -> $ccex, @eval (cc,cv,kx,ky,t) -> $cvex)
end

define_bzmask(sim::Simulation,::SingleMode) = identity

function define_observable_functions(sim::Simulation,solver::SingleMode)
    return [define_observable_functions(sim,solver,o) for o in sim.observables]
end

function define_observable_functions(sim::Simulation,::SingleMode,o::Observable)
    return [@eval (u,p,t) -> $ex for ex in buildobservable_vec_of_expr(sim,o)]
end

