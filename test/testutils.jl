using LoggingExtras
using Random
using TerminalLoggers

if !@isdefined(_DAMYSOS_TEST_LOGGER_SET)
    global_logger(TerminalLogger(stderr, Logging.Info))
    const _DAMYSOS_TEST_LOGGER_SET = true
end

if !@isdefined(_DAMYSOS_TEST_RNG_SEEDED)
    Random.seed!(1234)
    const _DAMYSOS_TEST_RNG_SEEDED = true
end

if !@isdefined(_DAMYSOS_TESTRESULTS_DIR)
    const _DAMYSOS_TESTRESULTS_DIR = begin
        if haskey(ENV, "DAMYSOS_TESTRESULTS_DIR") && !isempty(ENV["DAMYSOS_TESTRESULTS_DIR"])
            dir = ENV["DAMYSOS_TESTRESULTS_DIR"]
            mkpath(dir)
            dir
        else
            mktempdir()
        end
    end
end

testresults_dir() = _DAMYSOS_TESTRESULTS_DIR

datafile(name::AbstractString) = joinpath(@__DIR__, "data", name)
