module Damysos


using Unitful,Accessors,Trapz,DifferentialEquations,Interpolations,Plots,DSP,DataFrames,Random,CSV

export Hamiltonian,GappedDirac,getϵ,getdx_cc,getdx_cv,getdx_vc,getdx_vv,getdipoles_x,getvels_x
export getvx_cc,getvx_cv,getvx_vc,getvx_vv
export DrivingField,GaussianPulse,get_efield,get_vecpot
export NumericalParameters,NumericalParams2d,NumericalParams1d
export Simulation,Ensemble,getparams,parametersweep
export Observable,Velocity,Occupation,Timesteps,getnames_obs
export UnitScaling,semiclassical_interband_range
export run_simulation,run_simulation1d,run_simulation2d

abstract type SimulationComponent{T} end
abstract type Hamiltonian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end
abstract type Observable{T}             <: SimulationComponent{T} end

include("Simulation.jl")
include("Hamiltonians.jl")
include("DrivingFields.jl")
include("NumericalParams.jl")
include("Observables.jl")
include("Data.jl")
include("Plotting.jl")
include("Core.jl")
include("Utility.jl")

end
