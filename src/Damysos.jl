module Damysos

using Unitful,Accessors,Trapz,DifferentialEquations,Interpolations,CairoMakie
using DSP,DataFrames,Random,CSV,Formatting,Folds

export Hamiltonian,GappedDirac,scalegapped_dirac
export getϵ,getdx_cc,getdx_cv,getdx_vc,getdx_vv
export getdipoles_x,getvels_x
export getvx_cc,getvx_cv,getvx_vc,getvx_vv
export DrivingField,GaussianPulse,get_efieldx,get_vecpotx
export NumericalParameters,NumericalParams2d,NumericalParams1d,NumericalParams2dSlice
export Simulation,Ensemble,getparams,parametersweep
export Observable,Velocity,Occupation,Timesteps,getnames_obs
export UnitScaling,semiclassical_interband_range,maximum_k
export run_simulation!,run_simulation1d!,run_simulation2d!
export savemetadata,save,load,savedata,loaddata

abstract type SimulationComponent{T} end
abstract type Hamiltonian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end
abstract type Observable{T}             <: SimulationComponent{T} end

include("Simulation.jl")
include("Ensemble.jl")
include("Hamiltonians.jl")
include("DrivingFields.jl")
include("NumericalParams.jl")
include("Observables.jl")
include("Data.jl")
include("Plotting.jl")
include("Core.jl")
include("Utility.jl")

end
