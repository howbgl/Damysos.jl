# Convergence testing

```@meta
CurrentModule = Damysos
```

Every Damysos [`Simulation`](@ref) discretises time and k-space on finite grids. The physically
meaningful result is the one obtained in the limit of an infinitely fine (or infinitely large) grid,
so a numerical parameter — the time step `dt`, the k-spacing `dkx`/`dky`, or the integration cutoff
`kxmax`/`kymax` — must be refined until the observables of interest stop changing. A
[`ConvergenceTest`](@ref) automates that sweep: it repeatedly runs the simulation while varying one
parameter, Richardson-extrapolates the observables, and stops once a target tolerance is reached (or
a time/iteration budget is exhausted).

## Convergence methods

A [`ConvergenceTestMethod`](@ref) specifies *which* parameter is refined and *how* it changes each
iteration:

- [`PowerLawTest`](@ref)`(parameter, multiplier)` — multiplies `parameter` by `multiplier` each
  iteration. Use for parameters best refined geometrically, e.g. halving the time step with
  `PowerLawTest(:dt, 0.5)`.
- [`LinearTest`](@ref)`(parameter, shift)` — adds `shift` to `parameter` each iteration.
- [`ExtendKymaxTest`](@ref)`(method)` — a specialisation that extends the integration region in the
  ``k_y`` direction by *reusing* previously computed strips instead of recomputing the whole grid.
  It wraps a `PowerLawTest`/`LinearTest` on `:kymax` and currently requires a Gaussian pulse along
  ``k_x`` (`φ = 0`).

`parameter` is a `Symbol` naming a field of either the k-grid (e.g. `:dkx`, `:kxmax`, `:kymax`) or
the time grid (e.g. `:dt`, `:t0`).

## Running a test

Construct a [`ConvergenceTest`](@ref) from a starting [`Simulation`](@ref) and a solver, choosing
the method and the stopping criteria, then call [`run!`](@ref):

```julia
using Damysos

sim    = make_test_simulation_tiny()          # any Simulation
solver = LinearChunked(256)

test = ConvergenceTest(sim, solver;
    method        = PowerLawTest(:dt, 0.5),   # halve the time step each iteration
    atolgoal      = 1e-6,
    rtolgoal      = 1e-2,
    maxtime       = Inf,
    maxiterations = 20)

result = run!(test; filepath = "dt_convergence.hdf5")

successful_retcode(result)                    # true if it converged
```

`run!` returns a [`ConvergenceTestResult`](@ref) and, unless `savedata = false`, writes every
completed simulation plus the extrapolated result to the given `.hdf5` file. Use
[`successful_retcode`](@ref) to check whether the test reached its tolerance goal, and
[`terminated_retcode`](@ref) to check whether it stopped regularly (including hitting the
`maxtime`/`maxiterations` budget).

The stopping criteria interact as follows: the test ends as soon as **either** the extrapolated
observables satisfy both `atolgoal` and `rtolgoal`, **or** `maxiterations` simulations have run,
**or** `maxtime` seconds have elapsed. `maxtime` accepts a plain number of seconds or a Unitful time
(e.g. `u"60minute"`).

## Resuming from disk

Because each iteration is saved, a test can be reconstructed from its `.hdf5` file and continued —
for example on a different machine or with a different solver:

```julia
test = ConvergenceTest("dt_convergence.hdf5", LinearCUDA())
run!(test)
```

See the [Data I/O](data.md) page for more on the on-disk format.

## API

```@docs; canonical=false
ConvergenceTest
PowerLawTest
LinearTest
ExtendKymaxTest
ConvergenceTestResult
successful_retcode
terminated_retcode
```
```
