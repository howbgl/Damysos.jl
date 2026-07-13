# Damysos

```@meta
CurrentModule = Damysos
```

*A package to solve the Semiconductor Bloch equations using GPU or CPU.*

Documentation for [Damysos](https://github.com/howbgl/Damysos.jl.git).

## Package features

- Modular building blocks (driving field, Hamiltonian etc. ) for creating simulations, see [Hamiltonian models](hamiltonians.md) for the available Hamiltonians
- Using velocity-gauge to parallelize over k-points
- Run any Simulation on GPU or CPU without re-writing any code
- Built using the powerful [DifferentialEquations.jl](https://github.com/SciML/DiffEqDocs.jl) and [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl)

## Manual

- [Getting started](tutorial.md) — build and run your first simulation.
- [Solvers](solvers.md) — choosing between the CPU and GPU backends.
- [Convergence testing](convergence.md) — automatically refine numerical parameters.
- [Data I/O](data.md) — save, load, and reproduce published data.
- [Two-band formalism](twoband.md) and [Hamiltonian models](hamiltonians.md) — the physics and the available models.
- [Testing & development](testing.md) — the test tiers and CI workflows.

## Semiconductor Bloch equations

The computational task for this package consists of solving the Semiconductor Bloch equations (SBEs):

```math
[i\hbar\frac{\partial}{\partial t} + \epsilon_{mn}(\bm{k}_t) + \frac{i(1-\delta_{mn})}{T_2}]\rho_{mn}(\bm{k},t) = \sum_r \bm{E}(t)\cdot[\rho_{mr}(\bm{k},t)\bm{d}_{rn}(\bm{k}_t)-\bm{d}_{mr}(\bm{k}_t)\rho_{rn}(\bm{k},t)]
```

```math
\bm{k}_t=\bm{k}-\frac{e}{\hbar}\bm{A}(t)
```

By using the Coulomb/velocity gauge here ($\bm{E}(t)=-\dot{\bm{A}}(t)$), the SBEs decouple w.r.t $\bm{k}$ and are thus reduced to an ensemble of ODEs for different parameters.
See e.g. [Linberg et al](https://doi.org/10.1103/PhysRevB.38.3342), [Wikipedia](https://en.wikipedia.org/w/index.php?title=Semiconductor_Bloch_equations&oldid=1215737751), [Wilhelm et al](https://doi.org/10.1103/PhysRevB.103.125419) for more detailed description of SBEs.

## Integrators & solvers

The time-propagation is implemented using solvers from [DifferentialEquations.jl](https://github.com/SciML/DiffEqDocs.jl) and [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl). Any solver from these packages can be used.

There are three solvers to choose from:

- [`LinearCUDA`](@ref): sum with linear indexing on (CUDA) GPU
- [`LinearChunked`](@ref): sum with linear indexing on CPU
- [`SingleMode`](@ref): propagates single k-point
