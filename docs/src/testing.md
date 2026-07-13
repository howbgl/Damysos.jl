# Testing & development

```@meta
CurrentModule = Damysos
```

Damysos' test suite is split into tiers so that everyday CI stays fast while the full physical
validation remains available on demand. This page describes how to run the tiers locally and how
they map onto the project's continuous-integration workflows.

## Test tiers

| Tier | Contents | Requirements |
|------|----------|--------------|
| `fast` (default) | small deterministic checks and smoke tests | none |
| `slow` | CPU regression tests with full simulations | multithreading recommended |
| `gpu` | CUDA smoke tests | a functional CUDA GPU |
| `full` | long-running convergence and multi-GPU validation | GPU(s); long runtime |

Selecting `full` (or `all`) implies every other tier.

## Running the tests

From the package directory, start Julia with multiple threads (recommended for the `slow` and
`full` tiers):

```
julia --project -t auto
```

and select tiers through `test_args`:

```julia
using Pkg

Pkg.test("Damysos")                              # fast only (default)
Pkg.test("Damysos"; test_args=["slow"])
Pkg.test("Damysos"; test_args=["gpu"])
Pkg.test("Damysos"; test_args=["slow", "gpu"])
Pkg.test("Damysos"; test_args=["full"])          # everything
```

Tests write their output (HDF5 files, plots) to a temporary directory by default. To keep the
results, point `DAMYSOS_TESTRESULTS_DIR` at a directory of your choice:

```
DAMYSOS_TESTRESULTS_DIR=/path/to/results julia --project -t auto -e 'using Pkg; Pkg.test(test_args=["slow"])'
```

## Continuous integration

The tiers map onto the workflows in [`.github/workflows/`](https://github.com/howbgl/Damysos.jl/tree/main/.github/workflows):

| Workflow | Trigger | Runs |
|----------|---------|------|
| `ci.yml` | every push / pull request | `fast` tier on Julia 1.10 and 1.11 |
| `slow-regression.yml` | nightly cron + manual | `slow` tier |
| `gpu.yml` | manual dispatch | `gpu` tier on a self-hosted GPU runner |
| `full-validation.yml` | manual dispatch | `full` tier |
| `documentation.yml` | push to `main`, tags, PRs | builds and deploys the docs |
| `CompatHelper.yml` | daily cron | opens PRs bumping `[compat]` bounds |
| `TagBot.yml` | release events | creates git tags for registered releases |

Because only the `fast` tier runs on every push, run the heavier tiers locally before opening a pull
request that touches solver internals, grids, or the convergence machinery.

## Building the documentation locally

The documentation is built with [Documenter.jl](https://documenter.juliadocs.org/). To render it
locally:

```
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/make.jl")'
```

The generated site is written to `docs/build/`.
```
