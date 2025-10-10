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
using DataStructures
using DSP
using EnumX
using HDF5
using Infiltrator
using Interpolations
using LinearAlgebra
using ProgressLogging
using Random
using Reexport
using Richardson
using TerminalLoggers

@reexport using DifferentialEquations
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

abstract type DamysosSolver end
abstract type Observable{T} end
abstract type SimulationComponent{T} end
abstract type Hamiltonian{T} end
abstract type KGrid{T} end
abstract type TimeGrid{T} end
abstract type Liouvillian{T}            <: SimulationComponent{T} end
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

include("Liouvillian.jl")
include("drivingfields/DrivingFields.jl")
include("grids/symmetric_cartesian_kgrids.jl")


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
include("odesolve_gpu.jl")
include("Core.jl")

end
