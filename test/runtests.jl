using Test

include(joinpath(@__DIR__, "testutils.jl"))

function parse_groups(args::Vector{String})
    groups = isempty(args) ? Set(["fast"]) : Set(lowercase.(args))
    if "all" in groups || "full" in groups
        union!(groups, ["fast", "slow", "gpu", "full"])
    end
    return groups
end

const TEST_GROUPS = parse_groups(ARGS)
@info "Damysos test groups" groups = sort(collect(TEST_GROUPS))

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
