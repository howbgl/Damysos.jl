using DiffEqGPU, DifferentialEquations, StaticArrays, CUDA, KernelAbstractions, Adapt

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


#################### complete version: featureset of vectorized_solve ####################


@kernel function my_solve_kernel(@Const(probs), alg, _us, _ts, _integs, dt, callback,
        tstops, nsteps,
        saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # @cushow typeof(@inbounds _integs[i])
    
    # get the actual problem for this thread
    prob = @inbounds probs[i]
    
    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)
    
    _saveat = get(prob.kwargs, :saveat, nothing)
    
    saveat = _saveat === nothing ? saveat : _saveat
    
    integ = init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
    callback, save_everystep, saveat)

    u0 = prob.u0
    tspan = prob.tspan

    integ.cur_t = 0
    if saveat !== nothing
        integ.cur_t = 1
        if prob.tspan[1] == saveat[1]
            integ.cur_t += 1
            @inbounds us[1] = u0
        end
    else
        @inbounds ts[integ.step_idx] = prob.tspan[1]
        @inbounds us[integ.step_idx] = prob.u0
    end

    integ.step_idx += 1
    # FSAL
    while integ.t < tspan[2] && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = DiffEqGPU.step!(integ, ts, us)
        !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
    end
    if integ.t > tspan[2] && saveat === nothing
        ## Intepolate to tf
        @inbounds us[end] = integ(tspan[2])
        @inbounds ts[end] = tspan[2]
    end

    if saveat === nothing && !save_everystep
        @inbounds us[2] = integ.u
        @inbounds ts[2] = integ.t
    end

    _integs[i] = snapshot(integ)
end


function handle_allocations(probs, prob::ODEProblem, alg;
        dt, 
        saveat = nothing,
        save_everystep = true,
        callback = CallbackSet(nothing), 
        tstops = nothing,
        kwargs...)
    
    backend = maybe_prefer_blocks(get_backend(probs))
    timeseries = prob.tspan[1]:dt:prob.tspan[2]
    nsteps = length(timeseries)

    prob = convert(ImmutableODEProblem, prob)
    dt = convert(eltype(prob.tspan), dt)

    if saveat === nothing
        if save_everystep
            len = length(timeseries)
            if tstops !== nothing
                len += length(tstops) - count(x -> x in tstops, timeseries)
                nsteps += length(tstops) - count(x -> x in tstops, timeseries)
            end
        else
            len = 2
        end
        ts = allocate(backend, typeof(dt), (len, length(probs)))
        fill!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (len, length(probs)))
    else
        saveat = if saveat isa AbstractRange
            _saveat = range(convert(eltype(prob.tspan), first(saveat)),
                convert(eltype(prob.tspan), last(saveat)),
                length = length(saveat))
            convert(StepRangeLen{
                    eltype(_saveat),
                    eltype(_saveat),
                    eltype(_saveat),
                    eltype(_saveat) === Float32 ? Int32 : Int64,
                },
                _saveat)
        elseif saveat isa AbstractVector
            adapt(backend, convert.(eltype(prob.tspan), saveat))
        else
            _saveat = prob.tspan[1]:convert(eltype(prob.tspan), saveat):prob.tspan[end]
            convert(StepRangeLen{
                    eltype(_saveat),
                    eltype(_saveat),
                    eltype(_saveat),
                    eltype(_saveat) === Float32 ? Int32 : Int64,
                },
                _saveat)
        end
        ts = allocate(backend, typeof(dt), (length(saveat), length(probs)))
        fill!(ts, prob.tspan[1])
        us = allocate(backend, typeof(prob.u0), (length(saveat), length(probs)))
    end

    tstops = adapt(backend, tstops)

    @show saveat
    @show tstops
    @show save_everystep
    integ = init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, 
        tstops, callback, save_everystep, saveat)

    return backend, ts, us, integ, nsteps
end

function get_integ(probs, prob::ODEProblem, alg;
    dt, saveat = nothing,
    save_everystep = true,
    debug = false, callback = CallbackSet(nothing), tstops = nothing,
    kwargs...)

    backend, ts, us, integ, nsteps = handle_allocations(probs, prob, alg;
        dt = dt, saveat = saveat, save_everystep = save_everystep,
        callback = callback, tstops = tstops, kwargs...)

    return integ
end

function my_vectorized_solve(probs, prob::ODEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        debug = false, callback = CallbackSet(nothing), tstops = nothing,
        integrators = nothing, kwargs...)

    backend, ts, us, integ, nsteps = handle_allocations(probs, prob, alg;
        dt = dt, saveat = saveat, save_everystep = save_everystep,
        callback = callback, tstops = tstops, kwargs...)

    integs = allocate(backend, typeof(snapshot(integ)), (length(probs),))
    @show typeof(integs)
    @info "Building kernel"
    kernel = my_solve_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    @info "Compiling & launching kernel"
    kernel(probs, alg, us, ts, integs, dt, callback, tstops, nsteps, saveat,
        Val(save_everystep);
        ndrange = length(probs))

    ts, us, integs
end

#################### custom version: only save_everystep ####################

@kernel function solve_ode_kernel(@Const(probs), alg, _us, _ts, _integs, dt, callback)

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
        @cushow i, integ.step_idx, integ.t, len
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

@kernel function resume_solve_kernel(_us, _ts, _integs, tfinal)

    i               = @index(Global, Linear)
    integ           = @inbounds init(_integs[i])
    integ.retcode   = DiffEqBase.ReturnCode.Default
    integ.cur_t     = 0
    integ.step_idx  = 1
    len             = size(_ts, 1)
    ts              = @inbounds view(_ts, :, i)
    us              = @inbounds view(_us, :, i)
    
    # don't save initial state
    @cushow i,integ.u[1],integ.u[2],integ.u[3]

    while integ.step_idx < len + 1 && integ.retcode != DiffEqBase.ReturnCode.Terminated
        saved_in_cb = DiffEqGPU.step!(integ, ts, us)
        !saved_in_cb && DiffEqGPU.savevalues!(integ, ts, us)
        @cushow i, integ.step_idx, convert(Float64,integ.t)
    end
    if integ.t > tfinal
        ## Intepolate to tf
        @inbounds us[end] = integ(tfinal)
        @inbounds ts[end] = tfinal
    end

    if !isnothing(_integs)
        _integs[i] = snapshot(integ)
    end
end

function resume_solve(integrators, tfinal)

    backend    = get_backend(integrators)
    integ      = init(Array(integrators)[1])
    tfinal     = convert(typeof(integ.t), tfinal)
    timeseries = integ.t:integ.dt:tfinal
    len        = length(timeseries) - 1 # omit initial time integ.t
    ts         = allocate(backend, typeof(integ.dt), (len, length(integrators)))
    fill!(ts, integ.t)
    us         = allocate(backend, typeof(integ.u), (len, length(integrators)))

    kernel = resume_solve_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    kernel(us, ts, integrators, tfinal; ndrange = length(integrators))

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

function solve_odes_gpu(probs, prob::ODEProblem, alg, dt;
    callback = CallbackSet(nothing),
    initial_integrators = nothing, 
    kwargs...)

    resume              = !isnothing(initial_integrators)
    integ               = get_integ(probs, prob, alg, dt; callback = callback, kwargs...)
    timeseries          = prob.tspan[1]:dt:prob.tspan[2]
    backend, prob, dt   = adapt_odeproblem(probs, prob, alg, dt; kwargs...)

    len     = isnothing(initial_integrators) ? length(timeseries) : length(timeseries) -1
    ts      = allocate(backend, typeof(dt), (len, length(probs)))
    fill!(ts, prob.tspan[1])
    us      = allocate(backend, typeof(prob.u0), (len, length(probs)))
    integs  = if isnothing(initial_integrators)
        allocate(backend, typeof(snapshot(integ)), (length(probs),))
    else
        initial_integrators
    end

    kernel = solve_ode_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    kernel(probs, alg, us, ts, integs, dt, callback; ndrange = length(probs))

    ts, us, integs
end

################################ test it ################################

trajectories = 10

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0; 0.0; 0.0]
tspan = (0., 5.)
p = @SVector [10., 28., 8 / 3.]
prob = ODEProblem{false}(lorenz, u0, tspan, p)

pars = [(@SVector rand(Float64, 3)) .* p for i in 1:trajectories]

## Building different problems for different parameters
probs = map(1:trajectories) do i
    DiffEqGPU.make_prob_compatible(remake(prob, p = pars[i]))
end

## Move the arrays to the GPU
probs = cu(probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)

# Run once for compilation
@time CUDA.@sync ts, us = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1)

@time CUDA.@sync ts, us = DiffEqGPU.vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1)

## Run my version
define_snapshot(probs, prob, GPUTsit5(), 0.1)

@time CUDA.@sync myts, myus, integs = solve_odes_gpu(probs, prob, GPUTsit5(), 0.1)

@time CUDA.@sync myts, myus, integs = solve_odes_gpu(probs, prob, GPUTsit5(), 0.1)

# 1st time is bugged in original version, mine is correct
# last are equal to numerical precision, but somehow not exactly equal (both ts and us)
@show all(ts[2:end-1,:] .== myts[2:end-1,:])
@show all(us[2:end-1,:] .== myus[2:end-1,:])

# t = 0 to 2.5
prob1 = ODEProblem{false}(lorenz, u0, (0.0, 2.5), p)
probs1 = map(1:trajectories) do i
    DiffEqGPU.make_prob_compatible(remake(prob1, p = pars[i]))
end
probs1 = cu(probs1)
define_snapshot(probs1, prob1, GPUTsit5(), 0.1)
@time CUDA.@sync ts1, us1, integs1 = solve_odes_gpu(probs1, prob1, GPUTsit5(), 0.1)

# t = 2.5 to 5
# prob2 = ODEProblem{false}(lorenz, u0, (2.5, 5.0), p)
# probs2 = map(1:trajectories) do i
#     DiffEqGPU.make_prob_compatible(remake(prob2, p = pars[i]))
# end
# probs2 = cu(probs2)
# define_snapshot(probs2, prob2, GPUTsit5(), 0.1)
@time CUDA.@sync ts2, us2, integs2 = resume_solve(integs1, 5.0)

@show all(myts .== vcat(ts1, ts2))
@show all(myus .== vcat(us1, us2))

## Run my versiont
define_snapshot(probs, prob, GPUTsit5(); save_everystep = true, dt = 0.1)

@time CUDA.@sync ts2, us2, integs = my_vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1)

@time CUDA.@sync ts2, us2, integs = my_vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1);


############################### old approach symbolize ###############################

function symbolize(x::DataType)
    if isprimitivetype(x) || x == String
        return Symbol(x)
    else
        return Expr(:curly, nameof(x), symbolize.(x.parameters)...)
    end
end
symbolize(x::Number) = x
symbolize(x::TypeVar) = Symbol(x)

function bottomtype(T::UnionAll)
    cur_T = T
    while cur_T isa UnionAll
        cur_T = cur_T.body
    end
    return cur_T
end

function immutize(T)
    bt = bottomtype(T)
    fnames = fieldnames(bt)
    ftypes = [symbolize(fieldtype(bt,var)) for var in fnames]
    eval(Expr(:struct,false,
        Expr(:curly,Symbol("Immutable",nameof(bt)),Symbol.(bt.parameters)...),
        Expr(:block,[Expr(:(::),f,t) for (f,t) in zip(fnames,ftypes)]...)))
end

function define_immutable_struct(T::UnionAll)
    immutize(T)
    bt      = bottomtype(T)
    @eval function $(Symbol("Immutable",nameof(bt)))(x::$T)
        fields  = [getproperty(x, f) for f in fieldnames($T)]
        return $(Symbol("Immutable",nameof(T)))(fields...)
    end
end

