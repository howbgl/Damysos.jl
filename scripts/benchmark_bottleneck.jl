using BenchmarkTools
using CUDA
using Damysos
using Logging

include(joinpath(@__DIR__, "..", "test", "test_simulations.jl"))

const DEFAULT_SAMPLES = 10
const VANILLA_RUN! = Damysos.run!

function myrun!(
    sim::Simulation,
    functions::Damysos.SimulationFunctions,
    solver::DamysosSolver = LinearChunked();
    kwargs...)
    return invokelatest(Damysos.run!, sim, define_functions(sim, solver), solver; kwargs...)
end

function run_silently!(
    runner,
    sim::Simulation,
    functions::Damysos.SimulationFunctions,
    solver::DamysosSolver)

    return with_logger(NullLogger()) do
        runner(sim, functions, solver;
            savedata = false,
            saveplots = false,
            showinfo = false)
    end
end

function show_trial(label::AbstractString, trial)
    println(label)
    show(stdout, MIME"text/plain"(), trial)
    println("\n")
end

function benchmark_runner(
    runner,
    sim_template::Simulation,
    functions::Damysos.SimulationFunctions,
    solver::DamysosSolver;
    samples::Integer = DEFAULT_SAMPLES)

    sim = deepcopy(sim_template)
    run_silently!(runner, sim, functions, solver)

    bench = @benchmarkable run_silently!($runner, $sim, $functions, $solver)  evals = 1 samples = samples

    return run(bench)
end

function benchmark_solver(
    label::AbstractString,
    sim_template::Simulation,
    functions::Damysos.SimulationFunctions,
    solver::DamysosSolver;
    samples::Integer = DEFAULT_SAMPLES)

    println("
=== $(label) ===
")
    show_trial("run!", benchmark_runner(VANILLA_RUN!, sim_template, functions, solver;
        samples = samples))
    show_trial("myrun! (invokelatest bottleneck)", benchmark_runner(myrun!, sim_template, functions, solver;
        samples = samples))
end

const CPU_SIM = make_test_simulation_2d(; id = "sim2d_cpu_bench")
const CPU_SOLVER = LinearChunked()
const CPU_FUNCTIONS = define_functions(CPU_SIM, CPU_SOLVER)

const CUDA_AVAILABLE = CUDA.functional()
const GPU_SIM = CUDA_AVAILABLE ?
    make_test_simulation_2d(0.01, 0.5, 0.5, 175, 100; id = "sim2d_gpu_bench") : nothing
const GPU_SOLVER = CUDA_AVAILABLE ? LinearCUDA(60_000, GPUVern7(), 1) : nothing
const GPU_FUNCTIONS = CUDA_AVAILABLE ? define_functions(GPU_SIM, GPU_SOLVER) : nothing

if abspath(PROGRAM_FILE) == @__FILE__
    samples = isempty(ARGS) ? DEFAULT_SAMPLES : parse(Int, first(ARGS))

    benchmark_solver("Reference 2d / LinearChunked", CPU_SIM, CPU_FUNCTIONS, CPU_SOLVER;
        samples = samples)

    if CUDA_AVAILABLE
        benchmark_solver(
            "Reference 2d / LinearCUDA",
            GPU_SIM,
            GPU_FUNCTIONS,
            GPU_SOLVER;
            samples = samples)
    else
        @warn "Skipping LinearCUDA benchmark because CUDA.jl is not functional."
    end
end
