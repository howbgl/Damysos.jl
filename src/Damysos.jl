module Damysos


using Accessors
using ArgCheck
using CairoMakie
using ColorSchemes
using CSV
using CUDA
using Dates
using Distributed
using DataFrames
using DSP
using EnumX
using HDF5
using Interpolations
using LinearAlgebra
using ProgressLogging
using Random
using Reexport
using Richardson
using TerminalLoggers

@reexport using OrdinaryDiffEq
@reexport using DiffEqGPU
@reexport using Unitful
@reexport using SpecialFunctions
@reexport using StaticArrays

import Base.promote_rule

export DamysosSolver
export DrivingField
export Hamiltonian
export Liouvillian
export NGrid
export PreparedSimulation

"""
    DamysosSolver

Abstract supertype for the integration strategies that run a [`Simulation`](@ref), such as
[`LinearChunked`](@ref), [`LinearCUDA`](@ref) and [`SingleMode`](@ref).
"""
abstract type DamysosSolver end

"""
    Observable{T}

Abstract supertype for quantities computed during a [`Simulation`](@ref), such as
[`Velocity`](@ref) and [`Occupation`](@ref).
"""
abstract type Observable{T} end

abstract type SimulationComponent{T} end

"""
    Hamiltonian{T}

Abstract supertype for the Hamiltonians describing the electronic band structure, such as
[`GappedDirac`](@ref).
"""
abstract type Hamiltonian{T} end

"""
    KGrid{T}

Abstract supertype for k-space integration grids, such as [`CartesianKGrid2d`](@ref) or
[`KGrid0d`](@ref).
"""
abstract type KGrid{T} end

"""
    TimeGrid{T}

Abstract supertype for time integration grids, such as [`SymmetricTimeGrid`](@ref).
"""
abstract type TimeGrid{T} end

abstract type AperiodicKGrid{T}         <: KGrid{T} end
abstract type PeriodicKGrid{T}          <: KGrid{T} end

"""
    Liouvillian{T}

Abstract supertype for the equations of motion governing the density-matrix dynamics, such as
[`TwoBandDephasingLiouvillian`](@ref).
"""
abstract type Liouvillian{T}            <: SimulationComponent{T} end

"""
    DrivingField{T}

Abstract supertype for the driving fields, such as [`GaussianAPulse`](@ref).
"""
abstract type DrivingField{T}           <: SimulationComponent{T} end


include("Utility.jl")
include("general_hamiltonian/GeneralTwoBand.jl")
include("UnitScaling.jl")
include("grids/NGrid.jl")
include("Simulation.jl")
include("grids/SymmetricTimeGrid.jl")

include("hamiltonians/GappedDirac.jl")
include("hamiltonians/QuadraticToy.jl")
include("hamiltonians/BilayerToy.jl")
include("hamiltonians/MonolayerhBN.jl")
include("hamiltonians/SemiconductorToy1d.jl")

include("Liouvillian.jl")
include("drivingfields/DrivingFields.jl")
include("grids/aperiodic_kgrids.jl")
include("grids/periodic_kgrids.jl")


include("observables/observable_utils.jl")
include("observables/auxiliary_structs.jl")
include("observables/Velocity.jl")
include("observables/VelocityX.jl")
include("observables/Occupation.jl")
include("observables/DensityMatrixSnapshots.jl")

include("printdimless_params.jl")
include("equationsofmotion.jl")
include("autoconverge/structs.jl")
include("autoconverge/methods.jl")


include("data/data_utils.jl")
include("data/construct_type_from_dict.jl")
include("data/savedata_hdf5.jl")

include("Plotting.jl")
include("PreparedSimulation.jl")
include("Core.jl")

end
