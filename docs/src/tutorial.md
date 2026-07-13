# Getting started

```@meta
CurrentModule = Damysos
```

This page walks through building and running a single [`Simulation`](@ref) from scratch. A
`Simulation` bundles everything needed for a calculation: the physical system (a
[`Liouvillian`](@ref) and a [`DrivingField`](@ref)), the numerical grids in time and k-space, the
[`Observable`](@ref)s to compute, and the [`UnitScaling`](@ref) that links the internal
dimensionless units to SI units.

A complete, ready-to-run version of the example below is available in
[`scripts/demo.jl`](https://github.com/howbgl/Damysos.jl/blob/main/scripts/demo.jl).

## Building the physical system

The dynamics are governed by a `Liouvillian`, built from a [`Hamiltonian`](@ref) and dephasing
times, and a `DrivingField`. Here we use a [`GappedDirac`](@ref) Hamiltonian and a
[`GaussianAPulse`](@ref) vector-potential pulse:

```julia
julia> using Damysos

julia> df = GaussianAPulse(2.0, 2π, 1.3)
GaussianAPulse:
 σ: 2.0
 ν: 1.0
 ω: 6.283185307179586
 eE: 1.3
 φ: 0.0
 ħω: 6.283185307179586
 θ: 0.0

julia> h = GappedDirac(0.1)
GappedDirac:
 m: 0.1
 vF: 1.0

julia> l = TwoBandDephasingLiouvillian(h, Inf, Inf)
TwoBandDephasingLiouvillian(GappedDirac)
  Hamiltonian: GappedDirac
  m: 0.1
  vF: 1.0
 t1: Inf
 t2: Inf
```

All quantities above are in *scaled* (dimensionless) units, which are used for every numerical
calculation. See the [Hamiltonian models](hamiltonians.md) page for the other available
Hamiltonians.

## Unit scaling

The [`UnitScaling`](@ref) links the dimensionless quantities to SI units via a characteristic time
and length. It is built on [Unitful.jl](https://github.com/PainterQubits/Unitful.jl), so any unit
recognised by that package can be used.

```julia
julia> us = UnitScaling(u"1/10THz", u"5.5e5m/s" / u"10THz")
UnitScaling:
 timescale: 100.0 fs
 lengthscale: 55.0 nm
```

Here the period of a ``1\,``THz pulse (``T_0 = 100\,``fs) is the timescale and
``l_c = \frac{5.5\times10^5\,\mathrm{m/s}}{T_0}``, so that a scaled Fermi velocity of ``v_F = 1``
corresponds to ``5.5\times10^5\,``m/s. Built-in conversion helpers make this explicit:

```julia
julia> velocitySI(1.0, us)
550000.0 m s^-1

julia> velocityscaled(u"5.5e5m/s", us)
1.0
```

For a convenient way to derive a scaling directly from a driving frequency and Fermi velocity, see
[`scaledriving_frequency`](@ref), which is used in [`scripts/demo.jl`](https://github.com/howbgl/Damysos.jl/blob/main/scripts/demo.jl).

## Grids

The integration grid in time is a [`TimeGrid`](@ref); currently the only option is a
[`SymmetricTimeGrid`](@ref):

```julia
julia> tgrid = SymmetricTimeGrid(0.01, -5df.σ)
SymmetricTimeGrid:
  dt: 0.01
  t0: -10.0
```

Integration in k-space uses a subtype of [`KGrid`](@ref). The grid type encodes the dimensionality
of the simulation:

1. Single k-mode ``\Rightarrow`` [`KGrid0d`](@ref)
2. One-dimensional Fermi sea ``\Rightarrow`` [`CartesianKGrid1d`](@ref)
3. Two-dimensional Fermi sea ``\Rightarrow`` [`CartesianKGrid2d`](@ref)

Periodic (tight-binding) Hamiltonians require periodic k-grids instead; see
[Hamiltonian models](hamiltonians.md). The time and k-space grids are combined into an
[`NGrid`](@ref):

```julia
julia> kgrid = CartesianKGrid2d(1.0, 100.0, 1.0, 100.0)
CartesianKGrid2d:
  dkx: 1.0
  kxmax: 100.0
  dky: 1.0
  kymax: 100.0

julia> grid = NGrid(kgrid, tgrid)
NGrid:
CartesianKGrid2d:
  dkx: 1.0
  kxmax: 100.0
  dky: 1.0
  kymax: 100.0
 SymmetricTimeGrid:
  dt: 0.01
  t0: -10.0
```

## Observables

The quantities to be computed are given as a `Vector` of [`Observable`](@ref)s constructed from the
grid:

```julia
julia> obs = [Velocity(grid), Occupation(grid)]
2-element Vector{Observable{Float64}}:
 Velocity{Float64}(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
 Occupation{Float64}(Float64[])
```

## Assembling and running the simulation

With all components in hand we build the [`Simulation`](@ref):

```julia
julia> sim = Simulation(l, df, grid, obs, us, "simulation-name")
```

and run it. Passing a `Simulation` directly to [`run!`](@ref) lets you choose the solver via the
`solver` keyword:

```julia
julia> result = run!(sim;
           solver   = LinearChunked(2000),
           saveplots = false,
           savedata  = true)
```

The return value is `sim.observables`. See the [`run!`](@ref) docstring for the full list of
keyword arguments (`savedata`, `saveplots`, `savepath`, `showinfo`, `nan_limit`), and the
[Data I/O](data.md) page for how results are written to disk.

## Reusing functions with `PreparedSimulation`

For maximal performance — for example when running many simulations that share the same physical
model — pre-compile the numerical functions once into a [`PreparedSimulation`](@ref):

```julia
julia> const psim = PreparedSimulation(sim, LinearChunked());

julia> result = run!(psim)
```

!!! warning "World age"
    The call to `run!(psim)` must be more recent in world age than the definition of the
    `PreparedSimulation`, otherwise a `MethodError` is thrown. See the Julia manual on the
    [world age mechanism](https://docs.julialang.org/en/v1/manual/worldage/#The-World-Age-mechanism).
    When building and running inside the same function, `run!(sim; solver=...)` handles this for you
    via `invokelatest`.

## Next steps

- [Solvers](solvers.md) — choosing between CPU and GPU backends.
- [Convergence testing](convergence.md) — automatically refining a numerical parameter until an
  observable converges.
- [Data I/O](data.md) — saving and loading simulations, and reproducing published data.
```
