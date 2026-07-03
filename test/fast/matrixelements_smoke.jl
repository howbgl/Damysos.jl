using Damysos
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))

import Damysos.paulivector
import Damysos.adiabatic_melements_numeric
import Damysos.estimate_atol
import Damysos.jac

const KRANGE_SMOKE = -1:0.3:1

@testset "MatrixElements Smoke" begin
    h = GappedDirac(0.2)

    @testset "Pauli σx (coarse grid)" begin
        pauli = paulivector(h)[1]
        cv_sym = @eval (kx, ky) -> $(σx_cv(h))
        cv_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli, kx, ky)[1, 2]
        @test all(isapprox.(sample(cv_sym, KRANGE_SMOKE), sample(cv_num, KRANGE_SMOKE); 
            atol = estimate_atol(h), rtol = 1e-6, nans = true))
    end

    @testset "Jacobian (coarse grid)" begin
        fns = [@eval (kx, ky) -> $ex for ex in jac(h)]
        sym = (kx, ky) -> [j(kx, ky) for j in fns]
        @test check_jacobian(k -> hvec(h, k[1], k[2]), sym;
            krange = KRANGE_SMOKE, atol = estimate_atol(h), rtol = 1e-6)
    end

    # periodic models with a != 1 to catch missing lattice-constant factors
    for hp in (MonolayerhBN(0.2, 1.0, 2.5), SemiconductorToy1d(2.5, 6.0, 2.8))
        @testset "$(Damysos.getshortname(hp)) Jacobian (coarse grid)" begin
            fns = [@eval (kx, ky) -> $ex for ex in jac(hp)]
            sym = (kx, ky) -> [j(kx, ky) for j in fns]
            for f in (sym, (kx, ky) -> jac(hp, kx, ky), Damysos.getjac(hp))
                @test check_jacobian(k -> hvec(hp, k[1], k[2]), f;
                    krange = KRANGE_SMOKE, atol = estimate_atol(hp), rtol = 1e-6)
            end
        end
    end
end
