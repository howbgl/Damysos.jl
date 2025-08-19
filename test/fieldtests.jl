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
        GaussianEPulse(σ,f,strength,θ,φ),
        GaussianAPulseX(σ,f,strength,θ))
end


@testset "Driving fields" begin
    fdm      = central_fdm(5, 1)
    tsamples = -10:0.01:10
    atol     = 1e-12
    rtol     = 1e-8
    @testset "Hardcoded, dispatch and closure functions" begin
        alldrivingfields    = getall_drivingfields()
        alldrivingfield_fns = getfield_functions.(alldrivingfields)

        for fns in alldrivingfield_fns
            @test check_drivingfield_functions(fns...;atol = atol, rtol = rtol)
        end        
    end
    
    @testset "Derivatives" begin
        for df in getall_drivingfields()
            fx, fy = get_efieldx(df), get_efieldy(df)
            ax, ay = get_vecpotx(df), get_vecpoty(df)
            @test isapprox(-fx.(tsamples), fdm.(ax, tsamples), atol = atol, rtol = rtol)
            @test isapprox(-fy.(tsamples), fdm.(ay, tsamples), atol = atol, rtol = rtol)
        end
    end

    @testset "Composite driving field" begin
        df1 = getall_drivingfields()[1]
        df2 = getall_drivingfields()[1]

        f1  = [get_efieldx(df1), get_efieldy(df1)]
        a1  = [get_vecpotx(df1), get_vecpoty(df1)]
        f2  = [get_efieldx(df2), get_efieldy(df2)]
        a2  = [get_vecpotx(df2), get_vecpoty(df2)]

        fx_total = t -> c1 * f1[1](t) + c2 * f2[1](t)
        fy_total = t -> c1 * f1[2](t) + c2 * f2[2](t)
        ax_total = t -> c1 * a1[1](t) + c2 * a2[1](t)
        ay_total = t -> c1 * a1[2](t) + c2 * a2[2](t)

        c1,c2   = rand(2)
        df_comp = c1 * df1 + c2 * df2

        @testset "Simple composite" begin

            fx, fy, ax, ay = get_efieldx(df_comp), get_efieldy(df_comp), 
                                                get_vecpotx(df_comp), get_vecpoty(df_comp)

            @test isapprox(fx.(tsamples), fx_total.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(fy.(tsamples), fy_total.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(ax.(tsamples), ax_total.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(ay.(tsamples), ay_total.(tsamples), atol = atol, rtol = rtol)
            
        end
        @testset "Nested composite" begin
            c3,c4 = rand(2)
            df_comp_nested = c3*df_comp + c4 * df1
            fx_nested = get_efieldx(df_comp_nested)
            fy_nested = get_efieldy(df_comp_nested)
            ax_nested = get_vecpotx(df_comp_nested)
            ay_nested = get_vecpoty(df_comp_nested)

            fx_total_nested = t -> c3 * fx_total(t) + c4 * f1[1](t)
            fy_total_nested = t -> c3 * fy_total(t) + c4 * f1[2](t)
            ax_total_nested = t -> c3 * ax_total(t) + c4 * a1[1](t)
            ay_total_nested = t -> c3 * ay_total(t) + c4 * a1[2](t)

            @test isapprox(fx_nested.(tsamples), fx_total_nested.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(fy_nested.(tsamples), fy_total_nested.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(ax_nested.(tsamples), ax_total_nested.(tsamples), atol = atol, rtol = rtol)
            @test isapprox(ay_nested.(tsamples), ay_total_nested.(tsamples), atol = atol, rtol = rtol)
        end
    end
end