module Damysos


using Accessors
using CairoMakie
using ColorSchemes
using CSV
using CUDA
using Dates
using Distributed
using DataFrames
using DSP
using HDF5
using Interpolations
using ProgressLogging
using Random
using Reexport
using TerminalLoggers

@reexport using DifferentialEquations
@reexport using DiffEqGPU
@reexport using Unitful
@reexport using SpecialFunctions
@reexport using StaticArrays

import Base.promote_rule

export DrivingField
export Hamiltonian
export Liouvillian
export NumericalParameters

abstract type DamysosSolver end
abstract type Observable{T} end
abstract type SimulationComponent{T} end
abstract type Hamiltonian{T} end
abstract type Liouvillian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end

const DEFAULT_K_CHUNK_SIZE = 256


include("Utility.jl")
include("general_hamiltonian/GeneralTwoBand.jl")
include("UnitScaling.jl")
include("Simulation.jl")
include("Ensemble.jl")
include("hamiltonians/GappedDirac.jl")
include("hamiltonians/HexWarpDirac.jl")
include("Liouvillian.jl")
include("drivingfields/DrivingFields.jl")
include("NumericalParams.jl")
include("observables/Observables.jl")
include("equationsofmotion.jl")
include("ConvergenceTest.jl")
include("Data.jl")
include("Plotting.jl")
include("Core.jl")

end
