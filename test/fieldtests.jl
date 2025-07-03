using CUDA
using CSV
using Damysos
using DataFrames
using FiniteDifferences
using LoggingExtras
using TerminalLoggers
using Test


function getfield_functions(df::DrivingField)

    field_expressions = [efieldx(df),efieldy(df),vecpotx(df),vecpoty(df)]
    hardcoded_fns = [@eval t -> $ex for ex in field_expressions]
    dispatch_fns = [t -> efieldx(df,t),t->efieldy(df,t),t->vecpotx(df,t),t->vecpoty(df,t)]
    closure_fns = [get_efieldx(df),get_efieldy(df),get_vecpotx(df),get_vecpoty(df)]
    return (hardcoded_fns,dispatch_fns,closure_fns)
end

function check_drivingfield_functions(hardcoded_fns,dispatch_fns,closure_fns;
    tsamples=-10:0.01:10, atol=1e-12, rtol=1e-8)
    check1 = [isapprox(f1.(tsamples),f2.(tsamples), atol = atol, rtol = rtol) for (f1,f2) in 
        zip(hardcoded_fns,dispatch_fns)] |> all
    check2 = [isapprox(f1.(tsamples),f2.(tsamples), atol = atol, rtol = rtol) for (f1,f2) in 
        zip(closure_fns,dispatch_fns)] |> all
    return check1 && check2
end

function getall_drivingfields()
    σ = 2.0
    f = 1.0
    strength = 8.1
    θ = 2π * rand()
    φ = 2π * rand()
    return (
        GaussianAPulse(σ,f,strength,θ,φ),
        GaussianEPulse(σ,f,strength,θ,φ))
end


@testset "Driving fields" begin
    @testset "Hardcoded, dispatch and closure functions" begin
        alldrivingfields    = getall_drivingfields()
        alldrivingfield_fns = getfield_functions.(alldrivingfields)

        for fns in alldrivingfield_fns
            @test check_drivingfield_functions(fns...)
        end        
    end
    
    @testset "Derivatives" begin
        fdm      = central_fdm(5, 1)
        tsamples = -10:0.01:10
        atol     = 1e-12
        rtol     = 1e-8
        for df in getall_drivingfields()
            fx, fy = get_efieldx(df), get_efieldy(df)
            ax, ay = get_vecpotx(df), get_vecpoty(df)
            @test isapprox(-fx.(tsamples), fdm.(ax, tsamples), atol = atol, rtol = rtol)
            @test isapprox(-fy.(tsamples), fdm.(ay, tsamples), atol = atol, rtol = rtol)
        end
    end
end