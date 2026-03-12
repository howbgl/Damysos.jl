using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))

function make_test_simulation_2d(
	dt::Real = 0.01,
	dkx::Real = 1.0,
	dky::Real = 1.0,
	kxmax::Real = 175,
	kymax::Real = 100)

	vf     = u"4.3e5m/s"
	freq   = u"5THz"
	m      = u"20.0meV"
	emax   = u"0.1MV/cm"
	tcycle = uconvert(u"fs", 1 / freq) # 100 fs
	t2     = tcycle / 4             # 25 fs
	t1     = Inf * u"1s"
	σ      = u"800.0fs"

	# converged at
	# dt = 0.01
	# dkx = 1.0
	# dky = 1.0
	# kxmax = 175
	# kymax = 100

	us   = scaledriving_frequency(freq, vf)
	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)
	tgrid = SymmetricTimeGrid(dt, -5df.σ)
	kgrid = CartesianKGrid2d(dkx, kxmax, dky, kymax)
	grid = NGrid(kgrid,tgrid)
	obs  = [Velocity(grid), Occupation(grid)]

	id    = "sim2d_gpu"

	return Simulation(l, df, grid, obs, us, id)
end


function test_2d(v_ref::Velocity,sim::Simulation,fns,solver::DamysosSolver;
	atol = 1e-10,
	rtol = 1e-2)
    
    res = run!(sim, fns, solver; saveplots = false, 
		savepath = joinpath(testresults_dir(), Damysos.getname(sim)))
	v   = filter(o -> o isa Velocity,res)[1]
	return isapprox(v, v_ref, atol = atol, rtol = rtol)
end


const referencedata2d = DataFrame(CSV.File(datafile("referencedata.csv")))
const vref2d = Velocity(
	referencedata2d.vx,
	referencedata2d.vxintra,
	referencedata2d.vxinter,
	referencedata2d.vy,
	referencedata2d.vyintra,
	referencedata2d.vyinter)

const sim_2d = make_test_simulation_2d()


skipcuda = !(CUDA.functional())

skipcuda &&  @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

lincuda = skipcuda ? nothing : LinearCUDA(10_000,GPUVern7(),1)
const fns_2d_lincuda = skipcuda ? nothing : define_functions(sim_2d, lincuda)

@testset "Reference (2d)" begin
    # @testset "LinearChunked" begin
    #     @test test_1d(vref1d,sim_1d,fns_1d_linchunked,linchunked)
    # end
    @testset "LinearCUDA" begin
        @test test_2d(vref2d,sim_2d,fns_2d_lincuda,lincuda) skip = skipcuda
    end
end
