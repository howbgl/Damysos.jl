
import DiffEqGPU.init
import DiffEqGPU.ImmutableODEProblem
import DiffEqGPU.get_backend
import DiffEqGPU.maybe_prefer_blocks
import DiffEqGPU.vectorized_solve
import DiffEqGPU.ode_solve_kernel
import DiffEqGPU.step!
import DiffEqGPU.savevalues!
import DiffEqGPU


############################### snapshot definition ###############################

function define_snapshot(probs, prob::ODEProblem, alg;
    dt, saveat = nothing,
    save_everystep = true,
    debug = false, callback = CallbackSet(nothing), tstops = nothing,
    kwargs...)

    integ = get_integ(probs, prob, alg;
        dt = dt, saveat = saveat, save_everystep = save_everystep,
        callback = callback, tstops = tstops, kwargs...)

    return define_snapshot(integ)    
end

function define_snapshot(probs, prob::ODEProblem, alg, dt;
    callback = CallbackSet(nothing),
    kwargs...)

    integ = get_integ(probs, prob, alg, dt; callback = CallbackSet(nothing), kwargs...)

    return define_snapshot(integ)    
end

function define_snapshot(integ)
    fnames  = fieldnames(typeof(integ))
    ftypes  = fieldtypes(typeof(integ))
    name    = gensym("IntegSnapshot")
    eval(Expr(:struct,false,name,
        Expr(:block,(Expr(:(::),f,t) for (f,t) in zip(fnames,ftypes))...)))

    args = [Expr(:., :integ, QuoteNode(f)) for f in fnames]

    @eval function snapshot(integ::$(typeof(integ)))
        return $(name)($(args...))
    end

    args = [Expr(:., :snapshot, QuoteNode(f)) for f in fnames]

    @eval function init(snapshot::$(name))
        return $(typeof(integ))($(args...))
    end

    return eval(:($name))
end

#################### custom version: only save_everystep ####################

@kernel function solve_kernel(@Const(probs), alg, _us, _ts, _integs, dt, callback)

    i = @index(Global, Linear)
    
    # get the actual problem for this thread
    prob    = @inbounds probs[i]
    
    # get the input/output arrays for this thread
    ts      = @inbounds view(_ts, :, i)
    us      = @inbounds view(_us, :, i)
    tspan   = prob.tspan
    len     = length(ts)

    integ = init(alg, prob.f, false, prob.u0, tspan[1], dt, prob.p, 
        nothing, callback, true, nothing)

    integ.cur_t = 0
    @inbounds ts[integ.step_idx] = tspan[1]
    @inbounds us[integ.step_idx] = prob.u0
    integ.step_idx += 1

    # FSAL
    while integ.step_idx < len + 1 && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = DiffEqGPU.step!(integ, ts, us)
        !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
    end
    if integ.t > tspan[2]
        ## Intepolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if !isnothing(_integs)
        _integs[i] = snapshot(integ)
    end
end

@kernel function resume_solve_kernel(_us, _ts, _integs, tspan)

    i               = @index(Global, Linear)
    integ           = @inbounds init(_integs[i])
    integ.retcode   = DiffEqBase.ReturnCode.Default
    integ.cur_t     = 0
    integ.step_idx  = 1
    len             = size(_ts, 1)
    ts              = @inbounds view(_ts, :, i)
    us              = @inbounds view(_us, :, i)
    
    # don't save initial state

    while integ.step_idx < len + 1 && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = DiffEqGPU.step!(integ, ts, us)
        !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
    end
    if integ.t > tspan[2]
        ## Intepolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if !isnothing(_integs)
        _integs[i] = snapshot(integ)
    end
end

function resume_ode_solve(integrators, tspan)

    backend    = get_backend(integrators)
    integ      = init(Array(integrators)[1])
    tspan      = convert.(eltype(integ.dt), tspan)
    timeseries = tspan[1]:integ.dt:tspan[2]
    len        = length(timeseries) - 1 # omit initial time tspan[1]
    ts         = allocate(backend, typeof(integ.dt), (len, length(integrators)))
    fill!(ts, tspan[1])
    us         = allocate(backend, typeof(integ.u), (len, length(integrators)))

    kernel = resume_solve_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    kernel(us, ts, integrators, tspan; ndrange = length(integrators))

    ts, us, integrators
end

function adapt_odeproblem(probs, prob::Union{ODEProblem,ImmutableODEProblem}, alg, dt; 
    kwargs...)

    backend = maybe_prefer_blocks(get_backend(probs))
    prob    = convert(ImmutableODEProblem, prob)
    dt      = convert(eltype(prob.tspan), dt)

    return backend, prob, dt
end

function get_integ(probs, prob::Union{ODEProblem,ImmutableODEProblem}, alg, dt;
    callback = CallbackSet(nothing),
    kwargs...)

    _, prob, dt = adapt_odeproblem(probs, prob, alg, dt; kwargs...)

    return init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, 
        nothing, callback, true, nothing)
end

function vectorized_solve(probs, prob::ODEProblem, alg, dt;
    callback = CallbackSet(nothing),
    kwargs...)

    integ               = get_integ(probs, prob, alg, dt; callback = callback, kwargs...)
    timeseries          = prob.tspan[1]:dt:prob.tspan[2]
    backend, prob, dt   = adapt_odeproblem(probs, prob, alg, dt; kwargs...)

    len     = length(timeseries)
    ts      = allocate(backend, typeof(dt), (len, length(probs)))
    fill!(ts, prob.tspan[1])
    us      = allocate(backend, typeof(prob.u0), (len, length(probs)))
    integs  = allocate(backend, typeof(snapshot(integ)), (length(probs),))

    kernel = solve_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    kernel(probs, alg, us, ts, integs, dt, callback; ndrange = length(probs))

    ts, us, integs
end
