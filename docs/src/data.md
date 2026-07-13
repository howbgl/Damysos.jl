# Data I/O

```@meta
CurrentModule = Damysos
```

Damysos stores simulations and their results in [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
files. A saved file contains the full specification of a [`Simulation`](@ref) — driving field,
Hamiltonian/Liouvillian, grids, unit scaling — alongside the computed observables, so a run can be
reloaded, inspected, or reproduced later.

## Saving

By default [`run!`](@ref) saves both data and plots when a simulation finishes. This is controlled
by keyword arguments:

```julia
run!(sim;
    savedata = true,               # write observables + simulation to disk
    saveplots = true,              # write default plots
    savepath = "myrun")            # target directory (default: joinpath(pwd(), getname(sim)))
```

To save explicitly, outside of `run!`, use `savedata`:

```julia
savedata(sim)                      # writes to joinpath(pwd(), getname(sim))
savedata(sim, "path/to/output")    # writes to a chosen directory
```

The low-level writer `savedata_hdf5(object, parent)` serialises individual objects (simulations,
grids, driving fields, …) into an open `HDF5.File`/`HDF5.Group` and is used internally by the
functions above.

## Loading

`load_obj_hdf5` reconstructs a Damysos object from a file path, or from an already-open
`HDF5.File`/`HDF5.Group`:

```julia
using Damysos, HDF5

sim = load_obj_hdf5("myrun/simulation.hdf5")     # from a path

file = h5open("myrun/data.hdf5")                 # or from an open handle / subgroup
sim  = load_obj_hdf5(file["t2=0.2"])
close(file)
```

The set of reconstructable types (simulations, all grid types, Hamiltonians, driving fields,
observables, and convergence-test methods) is registered internally; older on-disk layouts are
handled transparently for backwards compatibility.

A [`ConvergenceTest`](@ref) is reloaded with its own constructor, which rebuilds the test from all
completed iterations stored in the file (see [Convergence testing](convergence.md)):

```julia
test = ConvergenceTest("dt_convergence.hdf5", LinearChunked())
```

## Reproducing published data

Data published with Damysos can be reloaded and re-run directly from the archived HDF5 files. The
following reproduces a figure from [10.1103/4gwm-9lpy](https://doi.org/10.1103/4gwm-9lpy) using the
dataset archived at [10.5281/zenodo.17828205](https://doi.org/10.5281/zenodo.17828205). Pick an
`.hdf5` file from that archive's `rawdata` folder and load the desired simulation:

```julia
using Damysos, HDF5

file       = h5open("rawdata/Fig2_data.hdf5")
simulation = load_obj_hdf5(file["t2=0.2"])
close(file)
```

Choose a solver — the CPU [`LinearChunked`](@ref) or the GPU [`LinearCUDA`](@ref) — wrap the
simulation in a [`PreparedSimulation`](@ref), and run it:

```julia
solver  = LinearChunked()
psim    = PreparedSimulation(simulation, solver)
results = run!(psim; savepath = "Fig2_rerun")
```

A complete, runnable script is provided at
[`scripts/published_calculations/reproduce.jl`](https://github.com/howbgl/Damysos.jl/blob/main/scripts/published_calculations/reproduce.jl).
```
