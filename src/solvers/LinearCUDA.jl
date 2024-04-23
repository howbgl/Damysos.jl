
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
    algorithm::DiffEqGPU.GPUODEAlgorithm=GPUTsit5())
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
    kchunk::Vector{<:SVector{2,<:Real}},
    us::CuArray{<:SVector{2,<:Complex}},
    ts::CuArray{<:Real,2},
    bzmask::Function,
    obsfuncs::Vector{Vector{Function}})

    d_kchunk    = cu(kchunk)
    us          .*= bzmask.(d_kchunk',ts)
end