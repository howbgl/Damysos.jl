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

    fns = define_functions(sim, solver)

    Damysos.prerun!(sim, solver; kwargs...)
    Base.invokelatest(Damysos._run!, sim, fns, solver)
    Damysos.postrun!(sim; kwargs...)

    return sim.observables
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

    bench = @benchmarkable run_silently!($runner, sim, $functions, $solver) setup = (
        sim = deepcopy($sim_template)
    ) evals = 1 samples = samples

    return run(bench)
end

function benchmark_solver(
    label::AbstractString,
    sim_template::Simulation,
    solver::DamysosSolver;
    samples::Integer = DEFAULT_SAMPLES)

    functions = define_functions(sim_template, solver)

    println("\n=== $(label) ===\n")
    show_trial("run!", benchmark_runner(VANILLA_RUN!, sim_template, functions, solver;
        samples = samples))
    show_trial("myrun!", benchmark_runner(myrun!, sim_template, functions, solver;
        samples = samples))
end

function main(samples::Integer = DEFAULT_SAMPLES)
    cpu_sim = make_test_simulation_1d(; id = "sim1d_cpu_bench")
    benchmark_solver("Reference 1d / LinearChunked", cpu_sim, LinearChunked();
        samples = samples)

    if CUDA.functional()
        gpu_sim = make_test_simulation_1d(; id = "sim1d_gpu_bench")
        benchmark_solver(
            "Reference 1d / LinearCUDA",
            gpu_sim,
            LinearCUDA(10_000, GPUVern7(), 1);
            samples = samples)
    else
        @warn "Skipping LinearCUDA benchmark because CUDA.jl is not functional."
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    samples = isempty(ARGS) ? DEFAULT_SAMPLES : parse(Int, first(ARGS))
    main(samples)
end
