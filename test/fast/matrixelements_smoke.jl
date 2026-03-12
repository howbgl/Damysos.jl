using Damysos
using FiniteDifferences
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))

import Damysos.paulivector
import Damysos.adiabatic_melements_numeric
import Damysos.estimate_atol
import Damysos.jac

const KRANGE = -1:0.3:1

sample(f, krange = KRANGE) = [f(kx, ky) for kx in krange, ky in krange]

function check_tensor_data(data1, data2; atol = 1e-8, rtol = 1e-6)
    ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)
    isanynan(x) = x isa AbstractArray ? any(isnan.(x)) : isnan(x)
    return all([any(isanynan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
end

function check_jacobian(fn, dfn; krange = KRANGE, atol = 1e-8, rtol = 1e-6)
    fd = central_fdm(5, 1)
    data = sample(dfn, krange)
    data_fdm = sample((kx, ky) -> jacobian(fd, fn, [kx, ky])[1], krange)
    return check_tensor_data(data, data_fdm; atol = atol, rtol = rtol)
end

@testset "MatrixElements Smoke" begin
    h = GappedDirac(0.2)

    @testset "Pauli σx (coarse grid)" begin
        pauli = paulivector(h)[1]
        cv_sym = @eval (kx, ky) -> $(σx_cv(h))
        cv_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli, kx, ky)[1, 2]
        @test all(isapprox.(sample(cv_sym), sample(cv_num); atol = estimate_atol(h), rtol = 1e-6, nans = true))
    end

    @testset "Jacobian (coarse grid)" begin
        fns = [@eval (kx, ky) -> $ex for ex in jac(h)]
        sym = (kx, ky) -> [j(kx, ky) for j in fns]
        @test check_jacobian(k -> hvec(h, k[1], k[2]), sym; atol = estimate_atol(h), rtol = 1e-6)
    end
end
