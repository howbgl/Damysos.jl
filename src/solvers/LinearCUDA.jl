
export LinearCUDA

"""
    LinearCUDA{T}

Represents an integration strategy for k-space via simple midpoint sum, where individual
k points are computed concurrently on a CUDA GPU

# Fields
- `kchunksize::T`: number of k-points in one chunk. Every task/worker gets one chunk. 
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

