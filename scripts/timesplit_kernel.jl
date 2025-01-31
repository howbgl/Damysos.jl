using DiffEqGPU, DifferentialEquations, StaticArrays, CUDA, KernelAbstractions, Adapt, Cthulhu

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


struct ImmutableGPUT5I{IIP, S, T, ST, P, F, TS, CB, AlgType} <: DiffEqBase.AbstractODEIntegrator{AlgType, IIP, S, T}
    alg::AlgType
    f::F                  # eom
    uprev::S              # previous state
    u::S                  # current state
    tmp::S                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    u_modified::Bool
    tstops::TS
    tstops_idx::Int
    callback::CB
    save_everystep::Bool
    saveat::ST
    cur_t::Int
    step_idx::Int
    event_last_time::Int
    vector_event_last_time::Int
    last_event_error::T
    k1::S                 #intepolants
    k2::S
    k3::S
    k4::S
    k5::S
    k6::S
    k7::S
    cs::SVector{6, T}     # ci factors cache: time coefficients
    as::SVector{21, T}    # aij factors cache: solution coefficients
    rs::SVector{22, T}    # rij factors cache: interpolation coefficients
    retcode::DiffEqBase.ReturnCode.T

end
function ImmutableGPUT5I(
    s::DiffEqGPU.GPUTsit5Integrator{IIP, S, T, ST, P, F, TS, CB, AlgType}) where {IIP, S, T, ST, P, F, TS, CB, AlgType}
    return ImmutableGPUT5I{IIP, S, T, ST, P, F, TS, CB, AlgType}(
        s.alg,
        s.f,
        s.uprev,
        s.u,
        s.tmp,
        s.tprev,
        s.t,
        s.t0,
        s.dt,
        s.tdir,
        s.p,
        s.u_modified,
        s.tstops,
        s.tstops_idx,
        s.callback,
        s.save_everystep,
        s.saveat,
        s.cur_t,
        s.step_idx,
        s.event_last_time,
        s.vector_event_last_time,
        s.last_event_error,
        s.k1,
        s.k2,
        s.k3,
        s.k4,
        s.k5,
        s.k6,
        s.k7,
        s.cs,
        s.as,
        s.rs,
        s.retcode)
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

    @cushow integ
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


function my_vectorized_solve(probs, prob::ODEProblem, alg;
        dt, saveat = nothing,
        save_everystep = true,
        debug = false, callback = CallbackSet(nothing), tstops = nothing,
        kwargs...)
    backend = get_backend(probs)
    backend = maybe_prefer_blocks(backend)
    # if saveat is specified, we'll use a vector of timestamps.
    # otherwise it's a matrix that may be different for each ODE.
    timeseries = prob.tspan[1]:dt:prob.tspan[2]
    nsteps = length(timeseries)

    prob = convert(ImmutableODEProblem, prob)
    dt = convert(eltype(prob.tspan), dt)

    if saveat === nothing
        if save_everystep
            len = length(prob.tspan[1]:dt:prob.tspan[2])
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
        tstops,  callback, save_everystep, saveat)
    # @show typeof(integ)
    # integs = allocate(backend, typeof(integ), (length(probs),))

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

