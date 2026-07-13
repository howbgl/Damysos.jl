[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://howbgl.github.io/Damysos.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://howbgl.github.io/Damysos.jl/stable)
[![Build Status](https://github.com/howbgl/Damysos.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/howbgl/Damysos.jl/actions)
[![Coverage Status](https://codecov.io/gh/howbgl/Damysos.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/howbgl/Damysos.jl)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18414810.svg)](https://doi.org/10.5281/zenodo.18414810)

Shield: [![CC BY-ND 4.0][cc-by-nd-shield]][cc-by-nd]

This work is licensed under a
[Creative Commons Attribution-NoDerivs 4.0 International License][cc-by-nd].

[![CC BY-ND 4.0][cc-by-nd-image]][cc-by-nd]

[cc-by-nd]: https://creativecommons.org/licenses/by-nd/4.0/
[cc-by-nd-image]: https://licensebuttons.net/l/by-nd/4.0/88x31.png
[cc-by-nd-shield]: https://img.shields.io/badge/License-CC%20BY--ND%204.0-lightgrey.svg

# Damysos.jl

**Damysos.jl** solves the semiconductor Bloch equations (SBEs) for light–matter interaction in
two-band models on either CPU or GPU. It provides modular building blocks — driving fields,
Hamiltonians, and grids — that assemble into a `Simulation`, and uses the velocity gauge to
parallelize the dynamics over k-points. The same simulation runs unchanged on CPU or CUDA GPU(s),
built on top of [DifferentialEquations.jl](https://github.com/SciML/DiffEqDocs.jl) and
[DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl).

📖 **[Full documentation](https://howbgl.github.io/Damysos.jl/dev)**

## Installation

First make sure you have [Julia](https://julialang.org/downloads/) installed and set up. Since
Damysos is not yet in the official Julia registry it has to be cloned locally. Go to the desired
folder and run

```
git clone https://github.com/howbgl/Damysos.jl.git
```

## Quick example

```julia
using Damysos

df    = GaussianAPulse(2.0, 2π, 1.3)                 # Gaussian vector-potential pulse
h     = GappedDirac(0.1)                             # gapped Dirac Hamiltonian
l     = TwoBandDephasingLiouvillian(h, Inf, Inf)     # dynamics + dephasing times
us    = UnitScaling(u"1/10THz", u"5.5e5m/s" / u"10THz")  # link scaled units to SI

tgrid = SymmetricTimeGrid(0.01, -5df.σ)              # time grid
kgrid = CartesianKGrid2d(1.0, 100.0, 1.0, 100.0)     # 2d k-space grid
grid  = NGrid(kgrid, tgrid)
obs   = [Velocity(grid), Occupation(grid)]           # observables to compute

sim   = Simulation(l, df, grid, obs, us, "my-simulation")

result = run!(sim; solver = LinearChunked(2000))
```

See the [Getting started](https://howbgl.github.io/Damysos.jl/dev/tutorial/) guide for a
step-by-step walkthrough, and [`scripts/demo.jl`](scripts/demo.jl) for a complete example with
logging and plotting.

## Documentation

- [Getting started](https://howbgl.github.io/Damysos.jl/dev/tutorial/) — build and run a simulation.
- [Solvers](https://howbgl.github.io/Damysos.jl/dev/solvers/) — CPU vs GPU backends.
- [Convergence testing](https://howbgl.github.io/Damysos.jl/dev/convergence/) — automatically refine numerical parameters.
- [Data I/O](https://howbgl.github.io/Damysos.jl/dev/data/) — saving, loading, and reproducing published data.
- [Two-band formalism](https://howbgl.github.io/Damysos.jl/dev/twoband/) & [Hamiltonian models](https://howbgl.github.io/Damysos.jl/dev/hamiltonians/) — the physics and available models.

## Reproducing published data

Data published with Damysos can be reloaded and re-run directly from the archived HDF5 files. See the
[Data I/O](https://howbgl.github.io/Damysos.jl/dev/data/#Reproducing-published-data) page for a
walkthrough (using the dataset at <https://doi.org/10.5281/zenodo.17828205>), and
[`scripts/published_calculations/reproduce.jl`](scripts/published_calculations/reproduce.jl) for a
ready-to-run script.

## Testing

Tests are tiered (`fast`, `slow`, `gpu`, `full`) to keep default CI fast while preserving full
validation. Run the default fast tier with `Pkg.test("Damysos")`; see the
[Testing & development](https://howbgl.github.io/Damysos.jl/dev/testing/) page for the other tiers
and the CI workflows.
