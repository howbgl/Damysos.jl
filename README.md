# Damysos

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://how09898.git-pages.uni-regensburg.de/Damysos.jl/dev/)
[![Build Status](https://git.uni-regensburg.de/how09898/Damysos.jl/badges/main/pipeline.svg)](https://git.uni-regensburg.de/how09898/Damysos.jl/pipelines)
[![Coverage](https://git.uni-regensburg.de/how09898/Damysos.jl/badges/main/coverage.svg)](https://git.uni-regensburg.de/how09898/Damysos.jl/commits/main)

## Installation

### Install julia
First make sure you have Julia installed and setup (https://julialang.org/downloads/).

### Clone package
Since Damysos is not yet in the official Julia registry it needs to be cloned locally. Go to the desired folder and run
```
git clone https://git.uni-regensburg.de/how09898/Damysos.jl.git
```

## Testing package (warning: long runtime possible)


Navigate to the package directory and open a Julia prompt via
```
julia --project -t auto
```
Using multiple threads via `-t auto` is recommended to improve runtime of the test simulations.
```julia
julia> using Damysos,Pkg

julia> Pkg.test("Damysos")
```
This runs a few test simulations both CPU and GPU (if available), which might take a long time depending on the machine.

## Usage

### Demo script

The script [demo.jl](scripts/demo.jl) shows how to run a single simulation and save the results as well as some automated plots.

### Setting up your own simulation

The `Simulation` object holds information about the physical system as well as numerical parameters.
The physical system is described by a `Liouvillian` and a `DrivingField` governing the dynamics.

```julia
julia> using Damysos

julia> df = GaussianAPulse(2.0,2π,1.3)
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

julia> l = TwoBandDephasingLiouvillian(h,Inf,Inf)
TwoBandDephasingLiouvillian(GappedDirac)
  Hamiltonian: GappedDirac
  m: 0.1
  vF: 1.0
 t1: Inf
 t2: Inf

```

All quantities above are in scaled units, which are used for all numerical calculations. The `UnitScaling` links these dimensionless quantities to SI units via a characteristic time and length.
```julia
julia> us = UnitScaling(u"1/10THz",u"5.5e5m/s" / u"10THz")
UnitScaling:
 timescale: 100.0 fs
 lengthscale: 55.0 nm
```
It is built on the [Unitful.jl](https://github.com/PainterQubits/Unitful.jl) package, thus all units recognized by said package can be used here.
The period of a $1\,$THz pulse ($T_0=100\,$fs) was chosen as timescale and $l_c=\frac{5.5\times10^5\frac{\mathrm{m}}{\mathrm{s}}}{T_0}$, such that a scaled Fermi velocity of $v_F=1$ corresponds to $5.5\times10^5$m/s.
We can check this via the builtin conversion methods:
```julia
julia> velocitySI(1.0,us)
550000.0 m s^-1

julia> velocityscaled(u"5.5e5m/s",us)
1.0
```

The observables to be calculated by the simulation are given via a Vector of `Observable` objects.
```julia
julia> obs = [Velocity(l),Occupation(l)]
2-element Vector{Observable{Float64}}:
 Velocity{Float64}(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
 Occupation{Float64}(Float64[])

```

The timegrid used for integration is specified via a `TimeGrid` object.
Momentarily there is only the option of a `SymmetricTimeGrid <: TimeGrid`:
```julia
julia> tgrid = SymmetricTimeGrid(0.01,-5df.σ)
SymmetricTimeGrid:
  dt: 0.01
  t0: -10.0
```
Integration in k-space is specified via a subtype of `KGrid`.
At the moment only cartesian grids and a single k-point are implemented, where the object is specific to the type of simulation:

1. Single k-mode => `KGrid0d`
2. One-dimensional Fermi sea => `CartesianKGrid1d`
3. Two-dimensional Fermi sea => `CartesianKGrid2d`

Time- and k-space grids form an `NGrid`:
```julia
julia> kgrid = CartesianKGrid2d(1.0,100,1,100)
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


With this we can create the `Simulation` object
```julia
julia> sim = Simulation(l,df,grid,obs,us,"simulation-name")
Simulation{Float64} (2d):
 TwoBandDephasingLiouvillian(GappedDirac)
   Hamiltonian: GappedDirac
   m: 0.1
   vF: 1.0
  t1: Inf
  t2: Inf
 GaussianAPulse:
   σ: 2.0
   ω: 6.283
   eE: 1.3
   φ: 0.0
   θ: 0.0
  
 NGrid:
 CartesianKGrid2d:
   dkx: 1.0
   kxmax: 100.0
   dky: 1.0
   kymax: 100.0
  SymmetricTimeGrid:
   dt: 0.01
   t0: -10.0
  
 Observables:
  Velocity
  Occupation
 UnitScaling:
  timescale: 100.0 fs
  lengthscale: 55.0 nm
 id: "simulation-name"
```

To run it, we need to choose a solver and get the specialized functions for it via `define_functions`:
```julia
julia> solver = LinearChunked(2000)
LinearChunked:
  - kchunksize: 2000
  - algorithm: EnsembleThreads()
  - odesolver: Vern7(; stage_limiter! = trivial_limiter!, step_limiter! = trivial_limiter!, thread = static(false), lazy = true,)

julia> fns = define_functions(sim,solver);

julia> res = run!(sim,fns,solver;savedata=true,saveplots=true,savedir="path/to/savedir")

```
