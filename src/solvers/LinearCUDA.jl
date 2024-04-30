
export LinearCUDA

"""
    LinearCUDA{T}

Represents an integration strategy for k-space via simple midpoint sum, where individual
k points are computed concurrently on a CUDA GPU

# Fields
- `kchunksize::T`: number of k-points in one concurrently executed chunk. 
- `algorithm::GPUODEAlgorithm`: algorithm for solving differential equations

# Examples
```jldoctest
julia> solver = LinearCUDA(256,GPUVern9())
LinearCUDA{Int64}(256, GPUVern9())
```
# See also
[`LinearChunked`](@ref LinearChunked), [`SingleMode`](@ref SingleMode)
"""
struct LinearCUDA{T<:Integer} <: DamysosSolver 
    kchunksize::T
    algorithm::DiffEqGPU.GPUODEAlgorithm
    function LinearCUDA{T}(
        kchunksize::T=DEFAULT_K_CHUNK_SIZE,
        algorithm::DiffEqGPU.GPUODEAlgorithm=GPUTsit5()) where {T}

        if CUDA.functional()
            new(kchunksize,algorithm)
        else
            throw(ErrorException(
                "CUDA.jl is not functional, cannot use LinearCUDA solver."))
        end
    end
end


function LinearCUDA(
    kchunksize::Integer=DEFAULT_K_CHUNK_SIZE,
    algorithm::DiffEqGPU.GPUODEAlgorithm=GPUVern7())
    return LinearCUDA{typeof(kchunksize)}(kchunksize,algorithm)
end

function Base.show(io::IO,::MIME"text/plain",s::LinearCUDA)
    println(io,"LinearCUDA:" |> escape_underscores)
    str = """
    - kchunksize: $(s.kchunksize)
    - algorithm: $(s.algorithm)
    """ |> escape_underscores
    print(io,prepend_spaces(str,2))
end


function run!(
    sim::Simulation,
    functions,
    solver::LinearCUDA;
    savedata=true,
    saveplots=true)

    prerun!(sim;savedata=savedata,saveplots=saveplots)

    @info """
        Solver: $(repr(solver))
    """
    rhs,bzmask,obsfuncs = functions
    totalobs            = Vector{Vector{Observable}}(undef,0)

    @progress name="Simulation" for ks in buildkgrid_chunks(sim,solver.kchunksize)
        d_ts,d_us = solvechunk(sim,solver,ks,rhs)
        obs       = deepcopy(sim.observables)

        sum_observables!(cu(ks),d_us,d_ts,bzmask,obs,obsfuncs)
        push!(totalobs,obs)
    end
    
    sim.observables .= sum(totalobs)

    postrun!(sim;savedata=savedata,saveplots=saveplots)

    return sim.observables 
end

define_rhs_x(sim::Simulation,::LinearCUDA) = @eval (u,p,t) -> $(buildrhs_x_expression(sim))

define_bzmask(sim::Simulation,::LinearCUDA) = define_bzmask(sim)

function define_observable_functions(sim::Simulation,solver::LinearCUDA)
    return [define_observable_functions(sim,solver,o) for o in sim.observables]
end

function define_observable_functions(sim::Simulation,::LinearCUDA,o::Observable)
    return [@eval (u,p,t) -> $ex for ex in buildobservable_vec_of_expr(sim,o)]
end

function solvechunk(
    sim::Simulation{T},
    solver::LinearCUDA,
    kchunk::Vector{SVector{2,T}},
    rhs::Function) where {T<:Real}

    prob = ODEProblem{false}(
        rhs,
        SA[zeros(Complex{T},2)...],
        gettspan(sim),
        kchunk[1])

    probs = map(kchunk) do k
        DiffEqGPU.make_prob_compatible(remake(prob,p=k))
    end
    probs = cu(probs)
    CUDA.@sync ts,us = DiffEqGPU.vectorized_solve(
        probs,
        prob,
        solver.algorithm;
        save_everystep=true,
        dt=sim.numericalparams.dt)

    return ts,us
end

function sum_observables!(
    d_kchunk::CuArray{<:SVector{2,<:Real}},
    d_us::CuArray{<:SVector{2,<:Complex}},
    d_ts::CuArray{<:Real,2},
    bzmask::Function,
    obs::Vector{<:Observable},
    obsfuncs::Vector{<:Vector})

    d_us        .*= bzmask.(d_kchunk',d_ts)
    buf         = zero(d_ts)
    return [sum_observables!(o,f,d_kchunk,d_us,d_ts,buf) for (o,f) in zip(obs,obsfuncs)]
end
