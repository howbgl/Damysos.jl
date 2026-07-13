# Solvers

```@meta
CurrentModule = Damysos
```

Damysos exploits the fact that, in the velocity gauge, the [semiconductor Bloch equations](index.md)
decouple in ``\bm{k}``: each k-point is an independent system of ODEs. A *solver* is the strategy
for distributing and integrating that ensemble of ODEs.
All solvers subtype `DamysosSolver` and are selected either through the `solver` keyword of
[`run!`](@ref) or when constructing a [`PreparedSimulation`](@ref).

The same `Simulation` runs unchanged on any compatible solver, so switching between CPU and GPU
requires no code changes.

## Choosing a solver

| Solver | Backend | Simulation dimension | Typical use |
|--------|---------|----------------------|-------------|
| [`LinearChunked`](@ref) | CPU (threads/processes) | 1d, 2d | Default; portable, no GPU required |
| [`LinearCUDA`](@ref) | CUDA GPU(s) | 1d, 2d | Large k-grids, best throughput |
| [`SingleMode`](@ref) | CPU | 0d (single k-point) | Inspecting one k-mode |

A few practical notes:

- **Start on the CPU.** [`LinearChunked`](@ref) needs no special hardware and is the default when
  no solver is given. For multithreading, start Julia with `-t auto`.
- **Scale up on the GPU.** [`LinearCUDA`](@ref) computes many k-points concurrently and is the right
  choice for large two-dimensional k-grids, provided a functional CUDA installation is available. It
  automatically spreads work across all visible GPUs unless `ngpus` is set.
- **`kchunksize` is the tuning knob.** Both linear solvers split the k-grid into chunks. Larger
  chunks improve throughput but use more memory; `LinearCUDA` will automatically subdivide the time
  axis into slices if a chunk would exceed available GPU memory (reported via a memory-estimate log
  message), so reduce `kchunksize` if you see that warning frequently.

## CPU: `LinearChunked`

```julia
julia> solver = LinearChunked()                     # defaults: 256 k-points/chunk, Vern7

julia> solver = LinearChunked(2000)                 # larger chunks

julia> solver = LinearChunked(2000, EnsembleThreads())  # explicit parallelisation
```

The parallelisation strategy defaults to `EnsembleThreads()` when Julia runs with multiple threads,
`EnsembleDistributed()` with multiple worker processes, and `EnsembleSerial()` otherwise.

```@docs; canonical=false
LinearChunked
```

## GPU: `LinearCUDA`

```julia
julia> solver = LinearCUDA()                    # all available GPUs, fixed-timestep GPUVern7

julia> solver = LinearCUDA(10_000, GPUVern7(), 1)   # 10⁴ k-points/chunk, single GPU

julia> solver = LinearCUDA(; rtol = 1e-8, atol = 1e-8)  # adaptive time-stepping
```

Passing `rtol`/`atol` switches the GPU integrator into adaptive mode; leaving them at `nothing`
(the default) uses a fixed timestep taken from the [`SymmetricTimeGrid`](@ref). `LinearCUDA`
requires `CUDA.functional()` to be `true`.

```@docs; canonical=false
LinearCUDA
```

## Single k-point: `SingleMode`

`SingleMode` integrates a single k-point and is used with a zero-dimensional
[`KGrid0d`](@ref) simulation.

```@docs; canonical=false
SingleMode
```
```
