# Two-band Hamiltonians

```@meta
CurrentModule = Damysos
```

Here we outline the implementation of calculating matrixelements of two-band Hamiltonians.

## Idea

Consider Hamiltonians defined in k-space which are of the general form

```math
\hat{H} = \vec{h}(\vec{k})\cdot\vec{\sigma}
```

Since this is a simple 2x2 matrix, one can express all desired operator matrix elements
in terms of $\vec{h}(\vec{k})$ and its derivatives.

Below we list the functions implemented so far.

## Eigenstate basis

At the moment all matrix elements are given in the basis of the Hamiltonian's eigenstates
describing valence- and conduction band states. **Some** matrix elements depend on the eigenstates' gauge, so we give the explicit form of the eigenstates used here in the following:

```math
\hat{H}\ket{\psi_\mu} = \epsilon_\mu\ket{\psi_\mu}
```

with

```math
\epsilon_{\mu}(\vec{k})\equiv\epsilon_{\pm}(\vec{k})=\pm\|\vec{h}(\vec{k})\|\equiv\pm h(\vec{k})
```

In $\sigma_z$-standardbasis ($\mu\in\{\uparrow,\downarrow\}$) the eigenstates read

```math
\braket{\mu|\psi_+} = \frac{1}{\sqrt{h(h+h_z)}}\begin{pmatrix}
    h + h_z\\h_x + i\,h
    _y
\end{pmatrix}
```

```math
\braket{\mu|\psi_-} = \frac{1}{\sqrt{h(h+h_z)}}\begin{pmatrix}
    -h_x + i\, h_y\\h + h_z
\end{pmatrix}
```

## Velocity operator matrix elements

The velocity operator is given by $\hat{v}^\mu=\frac{\partial\hat{H}}{\partial k_\mu}$ where $\mu\in\{x,y,z\}$.  We denote its matrix elements via

```math
v_{mn}^\mu = \braket{\psi_m|\hat{v}^\mu|\psi_n}
```

where we use the indices $m,n\in\{+,-\}=\{c,v\}$ for positive (negative) and conduction (valence) band interchangeably. The matrix elements can be simplified to

```math
\begin{aligned}
v_{cc}^\mu(\vec{k}) &= \frac{\partial\epsilon}{\partial k_\mu}\\
v_{cv}^\mu(\vec{k}) &= \frac{\partial\vec{h}}{\partial k_\mu}\cdot\vec{\sigma}_{cv}(\vec{k})\\
v_{vc}^\mu(\vec{k}) &=\left[v_{cv}^\mu(\vec{k})\right]^*\\
v_{vv}^\mu(\vec{k}) &= -\frac{\partial\epsilon(\vec{k})}{\partial k_\mu}\\
\end{aligned}
```

using the [eigenstates](#eigenstate-basis) and the matrix elements $\vec{\sigma}_{mn}=\braket{\psi_m|\vec{\sigma}|\psi_n}$ of the Pauli matrices.

## Dipole operator matrix elements

The dipole operator for a charge $q$ is given by $\hat{d}^\mu=q\,\hat{r}^\mu$ where $\mu\in\{x,y,z\}$. We denote its matrix elements via

```math
d_{mn}^\mu = q\braket{\psi_m|\hat{r}^\mu|\psi_n}
```

where we use the indices $m,n\in\{+,-\}=\{c,v\}$ for positive (negative) and conduction (valence) band interchangeably. The matrix elements can be simplified to

```math
\begin{aligned}
d_{cc}^\mu(\vec{k}) &= \frac{1}{2\epsilon(\vec{k})(\epsilon(\vec{k}) + h_z(\vec{k}))}
    \left( \frac{\partial\vec{h}}{\partial k_\mu} \right)^T\Omega\,\vec{h}(\vec{k})\\
d_{cv}^\mu(\vec{k}) &= \frac{-i}{\Delta\epsilon(\vec{k})}
    \frac{\partial\vec{h}(\vec{k})}{\partial k_\mu}\cdot\vec{\sigma}_{cv}(\vec{k})\\
d_{vc}^\mu(\vec{k}) &=\left[v_{cv}^\mu(\vec{k})\right]^*\\
d_{vv}^\mu(\vec{k}) &= -d_{cc}^\mu(\vec{k})
\end{aligned}
```

with the matrix

```math
\Omega = \begin{pmatrix}
    0 & -1 & 0\\
    1 & 0 & 0\\
    0 & 0 & 0
\end{pmatrix}.
```
