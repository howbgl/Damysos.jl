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
import Damysos.jac
import Damysos.getjac
import Damysos.dhdkx
import Damysos.getdhdkx
import Damysos.dhdky
import Damysos.getdhdky
import Damysos.gethvec

function sample(f, krange)
	return [f(kx, ky) for kx in krange, ky in krange]
end

function check_scalar(data1, data2; atol = 1e-12, rtol = 1e-12)
	ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)

	# skip any NaNs occuring at e.g. Dirac point
	return all([any(isnan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
end

function check_tensor(data1, data2; atol = 1e-12, rtol = 1e-12)
	ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)

	isanynan(x::AbstractArray) = any([isnan(x[i]) for i in eachindex(x)])

	# skip any NaNs occuring at e.g. Dirac point
	return all([any(isanynan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
end

function check_melements(a, b; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	check_scalar(sample(a, krange), sample(b, krange); atol = atol, rtol = rtol)
end


function check_dhdkm(fn, dfn, kindex; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	fd = central_fdm(5, 1)

	data     = sample(dfn, krange)
	data_fdm = sample((kx, ky) -> jacobian(fd, fn, [kx, ky])[1][:, kindex], krange)

	return check_tensor(data, data_fdm; atol = atol, rtol = rtol)
end



function check_jacobian(fn, dfn; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	fd       = central_fdm(5, 1)

	data     = sample(dfn,krange)
	data_fdm = sample((kx,ky)->jacobian(fd, fn, [kx, ky])[1],krange)

	return check_tensor(data, data_fdm; atol = atol, rtol = rtol)
end

function check_pauli_melements(h::GeneralTwoBand{T};
	krange = -1:0.02:1,
	atol = estimate_atol(h),
	rtol = 1e-12) where T <: Real

	pauli_symbolic = [(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)]
	check_results  = Bool[]

	for (pauli_symb, pauli_op) in zip(pauli_symbolic, paulivector(T))
		cv_symb, vc_symb = pauli_symb
		push!(check_results, check_melements(
			(kx, ky) -> cv_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[1, 2],
			krange = krange,
			atol = atol,
			rtol = rtol))
		push!(check_results, check_melements(
			(kx, ky) -> vc_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[2, 1],
			krange = krange,
			atol = atol,
			rtol = rtol))
	end
	return all(check_results)
end

const ALL_HAMILTONIANS = [GappedDirac(rand()), QuadraticToy(rand(2)...)]
const ALL_JACOBIANS    = [(kx, ky) -> @eval $(jac(h)) for h in ALL_HAMILTONIANS]

@testset "Pauli matrixelements" begin
	for h in ALL_HAMILTONIANS
		@testset "$(getshortname(h)) precompiled" begin
			for (pauli_expr, pauli_op) in zip(
				[(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)], paulivector(h))
				cv_sym = @eval (kx, ky) -> $(pauli_expr[1](h))
				vc_sym = @eval (kx, ky) -> $(pauli_expr[2](h))
				cv_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[1, 2]
				vc_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[2, 1]
				@test check_melements(cv_sym, cv_num; atol = estimate_atol(h))
				@test check_melements(vc_sym, vc_num; atol = estimate_atol(h))
			end
		end
		@testset "$(getshortname(h)) dispatch" begin
			@test check_pauli_melements(h)
		end
	end
end

@testset "Gradients & Jacobian" begin
	for h in ALL_HAMILTONIANS
		@testset "$(getshortname(h)) jacobians" begin
			fns = [@eval (kx, ky) -> $ex for ex in jac(h)]
			sym = (kx, ky) -> [j(kx, ky) for j in fns]

            for f in (sym,(kx,ky)->jac(h,kx,ky),getjac(h))
                @test check_jacobian(k -> hvec(h, k[1], k[2]),f;atol=estimate_atol(h))
            end
		end

		@testset "$(getshortname(h)) dhdkx" begin
			fns = [@eval (kx, ky) -> $ex for ex in dhdkx(h)]
			sym = (kx, ky) -> [f(kx, ky) for f in fns]

            for f in (sym,(kx, ky) -> dhdkx(h, kx, ky),getdhdkx(h))
                @test check_dhdkm(k -> hvec(h, k[1], k[2]),f,1;atol=estimate_atol(h))
            end
		end
	end
end

