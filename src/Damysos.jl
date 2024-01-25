module Damysos

using Unitful,Accessors,Trapz,DifferentialEquations,Interpolations,CairoMakie
using DSP,DataFrames,Random,CSV,Formatting,Distributed,Folds,FLoops,Dates,SpecialFunctions
using TerminalLoggers,ProgressLogging,ColorSchemes


export Hamiltonian,GappedDirac,scalegapped_dirac
export DrivingField,get_efieldx,get_vecpotx
export NumericalParameters,NumericalParams2d,NumericalParams1d,NumericalParams2dSlice
export run_simulation!,run_simulation1d!,runsim2d_kybatches!
export savemetadata,save,load,savedata,loaddata
export getvx_cc,getvx_cv,getvx_vc,getvx_vv
export getϵ,getdx_cc,getdx_cv,getdx_vc,getdx_vv

abstract type Observable{T} end
abstract type SimulationComponent{T} end
abstract type Hamiltonian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end

include("Simulation.jl")
include("Ensemble.jl")
include("Hamiltonians.jl")
include("drivingfields/DrivingFields.jl")
include("NumericalParams.jl")
include("observables/Observables.jl")
include("Data.jl")
include("Plotting.jl")
include("Core.jl")
include("Utility.jl")

end
