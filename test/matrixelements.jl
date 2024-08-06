using CUDA
using CSV
using Damysos
using DataFrames
using FiniteDifferences
using LoggingExtras
using TerminalLoggers
using Test

import Damysos.paulivector
import Damysos.adiabatic_melements_numeric
import Damysos.getshortname
import Damysos.estimate_atol



function check_melements(a, b; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	a_data = [a(kx, ky) for kx in krange, ky in krange]
	b_data = [b(kx, ky) for kx in krange, ky in krange]

    # skip any NaNs occuring at e.g. Dirac point
    ia(a,b) = Base.isapprox(a,b,atol=atol,rtol=rtol)
    
    return all([any(isnan.([a,b])) ? true : ia(a,b) for (a,b) in zip(a_data,b_data)])
end

function check_pauli_melements(h::GeneralTwoBand{T}; 
    krange = -1:0.02:1, 
    atol = estimate_atol(h), 
    rtol = 1e-12) where T <: Real

	pauli_symbolic = [(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)]
    check_results  = Bool[]

	for (pauli_symb, pauli_op) in zip(pauli_symbolic, paulivector(T))
		cv_symb, vc_symb = pauli_symb
		push!(check_results,check_melements(
			(kx, ky) -> cv_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[1, 2],
			krange = krange,
            atol = atol,
            rtol = rtol))
		push!(check_results,check_melements(
			(kx, ky) -> vc_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[2, 1],
			krange = krange,
            atol = atol,
            rtol = rtol))
	end
    return all(check_results)
end

const ALL_HAMILTONIANS = [GappedDirac(rand()),QuadraticToy(rand(2)...)]

const fd = central_fdm(5,1)

@testset "Pauli matrixelements" begin
    for h in ALL_HAMILTONIANS
        @testset "$(getshortname(h)) precompiled" begin
            for (pauli_expr,pauli_op) in zip(
                [(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)],paulivector(h)) 
                cv_sym = @eval (kx,ky) -> $(pauli_expr[1](h))
                vc_sym = @eval (kx,ky) -> $(pauli_expr[2](h))
                cv_num = (kx,ky) -> adiabatic_melements_numeric(h,pauli_op,kx,ky)[1,2]
                vc_num = (kx,ky) -> adiabatic_melements_numeric(h,pauli_op,kx,ky)[2,1]
                @test check_melements(cv_sym,cv_num;atol=estimate_atol(h))
                @test check_melements(vc_sym,vc_num;atol=estimate_atol(h))
            end
        end
        @testset "$(getshortname(h)) dispatch" begin
            @test check_pauli_melements(h)
        end
    end
end
