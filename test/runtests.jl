using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test


rm("testresults/", force = true, recursive = true)

global_logger(TerminalLogger(stderr, Logging.Info))


@testset "Damysos.jl" begin

	include("fieldtests.jl")
	include("matrixelements.jl")
	include("reference1d.jl")

	
end

rm("testresults/", force = true, recursive = true)
