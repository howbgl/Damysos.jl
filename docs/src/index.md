```@meta
CurrentModule = Damysos
```

# Damysos

*A package to solve the Semiconductor Bloch equations using GPU or CPU.*

Documentation for [Damysos](https://git.uni-regensburg.de/how09898/Damysos.jl).

# Package features
- Modular building blocks (driving field, Hamiltonian etc. ) for creating simulations
- Using velocity-gauge to parallelize over k-points
- Run any Simulation on GPU or CPU without re-writing any code
- Built using the powerful [DifferentialEquations.jl](https://github.com/SciML/DiffEqDocs.jl) and [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl)

# Semiconductor Bloch equations

The computational task for this package consists of solving the Semiconductor Bloch equations (SBEs):
$$
 [i\hbar\frac{\partial}{\partial t} + \epsilon_{mn}(\bm{k}_t) + \frac{i(1-\delta_{mn})}{T_2}]\rho_{mn}(\bm{k},t) = \sum_r \bm{E}(t)\cdot[\rho_{mr}(\bm{k},t)\bm{d}_{rn}(\bm{k}_t)-\bm{d}_{mr}(\bm{k}_t)\rho_{rn}(\bm{k},t)]
$$

$$\bm{k}_t=\bm{k}-\frac{e}{\hbar}\bm{A}(t)$$

By using the Coulomb/velocity gauge here ($\bm{E}(t)=-\dot{\bm{A}}(t)$), the SBEs decouple w.r.t $\bm{k}$ and are thus reduced to an ensemble of ODEs for different parameters ($\bm{k}$).
