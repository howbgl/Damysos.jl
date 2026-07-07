# Hamiltonian models

```@meta
CurrentModule = Damysos
```

This page lists all concrete Hamiltonians implemented in Damysos. Every model is a
two-band Hamiltonian of the form ``\hat{H} = \vec{h}(\vec{k})\cdot\vec{\sigma}``
subtyping [`GeneralTwoBand`](@ref); the machinery for eigenstates and velocity/dipole
matrix elements is described in the [two-band formalism](twoband.md).

```@docs; canonical=false
GeneralTwoBand
```

The models fall into two groups: *periodic* models defined on a lattice with a finite
Brillouin zone, and *nonperiodic* continuum models. The distinction matters when choosing
a k-space grid — the `Simulation` constructor enforces the compatible pairing.

## Periodic models

Periodic (tight-binding) models are defined on a lattice, so their Brillouin zone is
finite and the Hamiltonian is invariant under reciprocal lattice translations. They must
be paired with a periodic k-space grid sampling exactly one Brillouin zone, i.e.
[`CartesianMPKGrid1d`](@ref) or [`HexagonalMPKGrid2d`](@ref).

```@docs; canonical=false
MonolayerhBN
SemiconductorToy1d
```

## Nonperiodic models

Nonperiodic models describe a continuum approximation around a band extremum; k-space is
unbounded and the simulated region is chosen large enough for the physics at hand. They
are used with the aperiodic Cartesian k-grids such as [`CartesianKGrid1d`](@ref) or
[`CartesianKGrid2d`](@ref).

```@docs; canonical=false
GappedDirac
QuadraticToy
BilayerToy
```
