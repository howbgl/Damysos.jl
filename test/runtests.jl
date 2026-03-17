using CSV
using Damysos
using DataFrames
using Test

include(joinpath(@__DIR__, "testutils.jl"))

const referencedata1d = DataFrame(CSV.File(datafile("referencedata1d.csv")))
const vref1d = Velocity(
    referencedata1d.vx,
    referencedata1d.vxintra,
    referencedata1d.vxinter,
    referencedata1d.vy,
    referencedata1d.vyintra,
    referencedata1d.vyinter)

const referencedata2d = DataFrame(CSV.File(datafile("referencedata.csv")))
const vref2d = Velocity(
    referencedata2d.vx,
    referencedata2d.vxintra,
    referencedata2d.vxinter,
    referencedata2d.vy,
    referencedata2d.vyintra,
    referencedata2d.vyinter)

function parse_groups(args::Vector{String})
    groups = isempty(args) ? Set(["fast"]) : Set(lowercase.(args))
    if "all" in groups || "full" in groups
        union!(groups, ["fast", "slow", "gpu", "full"])
    end
    return groups
end

const TEST_GROUPS = parse_groups(ARGS)
@info "Damysos test groups" groups = sort(collect(TEST_GROUPS))
@info "Damysos tests run with showinfo = false; this suppresses the usual simulation startup info."

@testset "Damysos.jl" begin
    if "fast" in TEST_GROUPS
        @testset "fast" begin
            include(joinpath(@__DIR__, "fast", "fieldtests.jl"))
            include(joinpath(@__DIR__, "fast", "matrixelements_smoke.jl"))
            include(joinpath(@__DIR__, "fast", "simulation_smoke.jl"))
        end
    end

    if "slow" in TEST_GROUPS
        @testset "slow" begin
            include(joinpath(@__DIR__, "slow", "matrixelements.jl"))
            include(joinpath(@__DIR__, "slow", "reference1d_cpu.jl"))
            include(joinpath(@__DIR__, "slow", "snapshot_cpu.jl"))
            include(joinpath(@__DIR__, "slow", "compositefield_cpu.jl"))
        end
    end

    if "gpu" in TEST_GROUPS
        @testset "gpu" begin
            include(joinpath(@__DIR__, "gpu", "reference1d_gpu.jl"))
            include(joinpath(@__DIR__, "gpu", "reference2d_gpu.jl"))
            include(joinpath(@__DIR__, "gpu", "snapshot_gpu.jl"))
            include(joinpath(@__DIR__, "gpu", "compositefield_gpu.jl"))
        end
    end

    if "full" in TEST_GROUPS
        @testset "full" begin
            include(joinpath(@__DIR__, "full", "convergence_full.jl"))
            include(joinpath(@__DIR__, "full", "reference2d_multigpu.jl"))
        end
    end
end
