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

@kernel function my_solve_kernel(@Const(probs), alg, _us, _ts, dt, callback,
        tstops, nsteps,
        saveat, ::Val{save_everystep}) where {save_everystep}
    i = @index(Global, Linear)

    # get the actual problem for this thread
    prob = @inbounds probs[i]

    # get the input/output arrays for this thread
    ts = @inbounds view(_ts, :, i)
    us = @inbounds view(_us, :, i)

    _saveat = get(prob.kwargs, :saveat, nothing)

    saveat = _saveat === nothing ? saveat : _saveat

    integ = init(alg, prob.f, false, prob.u0, prob.tspan[1], dt, prob.p, tstops,
        callback, save_everystep, saveat)

    # @cushow integ
    # @cushow ismutable(integ)

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
end


function handle_allocations(probs, prob::ODEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        callback = CallbackSet(nothing), tstops = nothing,
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

    @info "Building kernel"
    kernel = my_solve_kernel(backend)

    if backend isa CPU
        @warn "Running the kernel on CPU"
    end

    @info "Compiling & launching kernel"
    kernel(probs, alg, us, ts, dt, callback, tstops, nsteps, saveat,
        Val(save_everystep);
        ndrange = length(probs))

    ts, us
end

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

function define_snapshot(integ)
    fnames  = fieldnames(typeof(integ))
    ftypes  = fieldtypes(typeof(integ))
    name    = gensym("Integ_Snapshot")
    eval(Expr(:struct,false,name,
        Expr(:block,[Expr(:(::),f,t) for (f,t) in zip(fnames,ftypes)]...)))

    @eval function snapshot(integ::$(typeof(integ)))
        return $(name)([getproperty(integ,n) for n in $fnames]...)
    end
    return eval(:($name))
end

trajectories = 10_000

function lorenz(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end

u0 = @SVector [1.0f0; 0.0f0; 0.0f0]
tspan = (0.0f0, 10.0f0)
p = @SVector [10.0f0, 28.0f0, 8 / 3.0f0]
prob = ODEProblem{false}(lorenz, u0, tspan, p)

## Building different problems for different parameters
probs = map(1:trajectories) do i
    DiffEqGPU.make_prob_compatible(remake(prob, p = (@SVector rand(Float32, 3)) .* p))
end

## Move the arrays to the GPU
probs = cu(probs)

## Finally use the lower API for faster solves! (Fixed time-stepping)

# Run once for compilation
@time CUDA.@sync ts, us = my_vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1f0)

@time CUDA.@sync ts, us = my_vectorized_solve(probs, prob, GPUTsit5();
    save_everystep = true, dt = 0.1f0);

