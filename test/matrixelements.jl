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
import Damysos.eigvecs_numeric

function sample(f, krange)
	return [f(kx, ky) for kx in krange, ky in krange]
end

function check_scalar_data(data1, data2; atol = 1e-12, rtol = 1e-12)
	ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)

	# skip any NaNs occuring at e.g. Dirac point
	return all([any(isnan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
end

function check_tensor_data(data1, data2; atol = 1e-12, rtol = 1e-12)
	ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)

	isanynan(x::AbstractArray) = any([isnan(x[i]) for i in eachindex(x)])

	# skip any NaNs occuring at e.g. Dirac point
	return all([any(isanynan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
end

function check_scalar(a, b; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	check_scalar_data(sample(a, krange), sample(b, krange); atol = atol, rtol = rtol)
end

function check_tensor(a, b; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	check_tensor_data(sample(a, krange), sample(b, krange); atol = atol, rtol = rtol)
end

function check_dhdkm(fn, dfn, kindex; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	fd = central_fdm(5, 1)

	data     = sample(dfn, krange)
	data_fdm = sample((kx, ky) -> jacobian(fd, fn, [kx, ky])[1][:, kindex], krange)

	return check_tensor_data(data, data_fdm; atol = atol, rtol = rtol)
end

function check_jacobian(fn, dfn; krange = -1:0.02:1, atol = 1e-12, rtol = 1e-12)
	fd       = central_fdm(5, 1)

	data     = sample(dfn,krange)
	data_fdm = sample((kx,ky)->jacobian(fd, fn, [kx, ky])[1],krange)

	return check_tensor_data(data, data_fdm; atol = atol, rtol = rtol)
end

function check_pauli_melements(h::GeneralTwoBand{T};
	krange = -1:0.02:1,
	atol = estimate_atol(h),
	rtol = 1e-12) where T <: Real

	pauli_symbolic = [(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)]
	check_results  = Bool[]

	for (pauli_symb, pauli_op) in zip(pauli_symbolic, paulivector(T))
		cv_symb, vc_symb = pauli_symb
		push!(check_results, check_scalar(
			(kx, ky) -> cv_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[1, 2],
			krange = krange,
			atol = atol,
			rtol = rtol))
		push!(check_results, check_scalar(
			(kx, ky) -> vc_symb(h, kx, ky),
			(kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[2, 1],
			krange = krange,
			atol = atol,
			rtol = rtol))
	end
	return all(check_results)
end

vx_op_fdm(h::GeneralTwoBand,kx,ky) = central_fdm(5,1)(x -> hmat(h,x,ky),kx)
vy_op_fdm(h::GeneralTwoBand,kx,ky) = central_fdm(5,1)(y -> hmat(h,kx,y),ky)

function getfunc_melements_dispatch(h::Hamiltonian,op::Symbol)
	return @eval (kx,ky) -> [
		$(Symbol(op,"_cc"))($h,kx,ky) 	$(Symbol(op,"_cv"))($h,kx,ky)
		$(Symbol(op,"_vc"))($h,kx,ky) 	$(Symbol(op,"_vv"))($h,kx,ky)]
end

function getfunc_melements_closure(h::Hamiltonian,op::Symbol)
	closures = @eval [
		$(Symbol("get",op,"_cc"))($h) 	$(Symbol("get",op,"_cv"))($h)
		$(Symbol("get",op,"_vc"))($h) 	$(Symbol("get",op,"_vv"))($h)]
				
	return (kx,ky) -> [f(kx,ky) for f in closures]
end

function getfunc_melements_eval(h::Hamiltonian,op::Symbol)
	evalfns = [@eval (kx,ky) -> $ex for ex in @eval [
		$(Symbol(op,"_cc"))($h) 	$(Symbol(op,"_cv"))($h)
		$(Symbol(op,"_vc"))($h) 	$(Symbol(op,"_vv"))($h)]]

	return (kx,ky) -> [f(kx,ky) for f in evalfns]
end

function deigvecs_dkx(h::GeneralTwoBand,kx,ky)
	fd = central_fdm(5,1)
	return fd(x -> eigvecs_numeric(h,x,ky),kx)
end

function deigvecs_dky(h::GeneralTwoBand,kx,ky)
	fd = central_fdm(5,1)
	return fd(y -> eigvecs_numeric(h,kx,y),ky)
end

hconj(x::AbstractMatrix) = transpose(conj(x))

const ALL_HAMILTONIANS = [GappedDirac(0.2), QuadraticToy(1.6,0.4), BilayerToy(1.0,0.6)]

@testset "Pauli matrixelements" begin
	for h in ALL_HAMILTONIANS
		@testset "$(getshortname(h)) precompiled" begin
			for (pauli_expr, pauli_op) in zip(
				[(σx_cv, σx_vc), (σy_cv, σy_vc), (σz_cv, σz_vc)], paulivector(h))
				cv_sym = @eval (kx, ky) -> $(pauli_expr[1](h))
				vc_sym = @eval (kx, ky) -> $(pauli_expr[2](h))
				cv_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[1, 2]
				vc_num = (kx, ky) -> adiabatic_melements_numeric(h, pauli_op, kx, ky)[2, 1]
				@test check_scalar(cv_sym, cv_num; atol = estimate_atol(h))
				@test check_scalar(vc_sym, vc_num; atol = estimate_atol(h))
			end
		end
		@testset "$(getshortname(h)) dispatch" begin
			@test check_pauli_melements(h)
		end
	end
end

# TODO make this more elegant/concise
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

        @testset "$(getshortname(h)) dhdky" begin
			fns = [@eval (kx, ky) -> $ex for ex in dhdky(h)]
			sym = (kx, ky) -> [f(kx, ky) for f in fns]

            for f in (sym,(kx, ky) -> dhdky(h, kx, ky),getdhdky(h))
                @test check_dhdkm(k -> hvec(h, k[1], k[2]),f,2;atol=estimate_atol(h))
            end
		end
	end
end

@testset "Velocity & Dipole matrixelements" begin
    for h in ALL_HAMILTONIANS
		fdiff_melems = [
			(kx,ky) -> adiabatic_melements_numeric(h, vx_op_fdm(h,kx,ky), kx, ky),
			(kx,ky) -> adiabatic_melements_numeric(h, vy_op_fdm(h,kx,ky), kx, ky),
			(kx,ky) -> im*hconj(eigvecs_numeric(h,kx,ky)) * deigvecs_dkx(h,kx,ky),
			(kx,ky) -> im*hconj(eigvecs_numeric(h,kx,ky)) * deigvecs_dky(h,kx,ky)
		]
        @testset "$(getshortname(h))" begin
			
			for (str,symb,fdm) in zip(["vx","vy","dx","dy"],[:vx,:vy,:dx,:dy],fdiff_melems)
				@testset "$str" begin
					fns = (
						getfunc_melements_closure(h,symb),
						getfunc_melements_dispatch(h,symb),
						getfunc_melements_eval(h,symb)
						)
					for fn in fns
						@test check_tensor(fn,fdm;atol = estimate_atol(h))
					end
				end
			end			
        end
    end
end