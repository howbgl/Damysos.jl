using CUDA
using CSV
using Damysos
using DataFrames
using HDF5
using LoggingExtras
using TerminalLoggers
using Test

import Damysos.load_obj_hdf5

function make_test_simulation_snap(
	dt::Real = 0.01,
	dkx::Real = 2.0,
	dky::Real = 1.0,
	kxmax::Real = 175,
	kymax::Real = 10)

	vf     = u"4.3e5m/s"
	freq   = u"5THz"
	m      = u"20.0meV"
	emax   = u"0.1MV/cm"
	tcycle = uconvert(u"fs", 1 / freq) # 100 fs
	t2     = tcycle / 4             # 25 fs
	t1     = Inf * u"1s"
	σ      = u"800.0fs"

	us   = scaledriving_frequency(freq, vf)
	h    = GappedDirac(energyscaled(m, us))
	l    = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
	df   = GaussianAPulse(us, σ, freq, emax)
	tgrid = SymmetricTimeGrid(dt, -5df.σ)
	kgrid = CartesianKGrid2d(dkx, kxmax, dky, kymax)
	grid = NGrid(kgrid,tgrid)
	ts   = collect(Damysos.gettsamples(tgrid))
	obs  = [DensityMatrixSnapshots(l,grid; tsamples = ts[1:3:end]), Occupation(grid)]

	id    = "snapshot_sim"

	return Simulation(l, df, grid, obs, us, id)
end

function test_snapshots_lichunked(sim::Simulation, fns, solver; atol=1e-10, rtol=1e-8)

	run!(sim, fns, solver; savedata=false, saveplots=false)
	
	ks 	= Damysos.getksamples(sim.grid.kgrid)
	dms = sim.observables[1]
	ts 	= dms.tsamples
	
	occ_ref = sim.observables[2]
	obs  	= Observable[Occupation(sim.grid)]

	bzmask = fns[2]

	for (i,dm) in enumerate(dms.density_matrices)
		cc 				= [real(m[1,1]) for m in dm.density_matrix]
		weights 		= bzmask.(ks, ts[i]) 
		obs[1].cbocc[i]	= sum(cc .* weights)
	end
	Damysos.applyweights_afterintegration!(obs, sim.grid.kgrid)
    Damysos.normalize!.(obs,(2π)^sim.dimensions)

	return isapprox(obs[1],occ_ref,atol=atol,rtol=rtol)
end

function test_snapshots_saving_loading(sim::Simulation)
	Damysos.savedata(sim,"testresults/snapshot_test")
	return isapprox(sim,Simulation("testresults/snapshot_test/data.hdf5"))
end

const sim_snap = make_test_simulation_snap()

linchunked = LinearChunked(256)
const fns_sim_snap_linchunked = define_functions(sim_snap, linchunked)

skipcuda = !(CUDA.functional())

skipcuda &&  @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."
lincuda = skipcuda ? nothing : LinearCUDA(10_000)
const fns_sim_snap_lincuda = skipcuda ? nothing : define_functions(sim_snap, lincuda)

@testset "DensityMatrixSnapshots" begin
    @testset "LinearChunked" begin
        @test test_snapshots_lichunked(sim_snap, fns_sim_snap_linchunked, linchunked)
    end
    @testset "LinearCUDA" begin
        @test test_snapshots_lichunked(sim_snap, fns_sim_snap_lincuda, lincuda) skip=skipcuda
    end
	@testset "Saving & loading" begin
		@test test_snapshots_saving_loading(sims_snap)
	end
end

