using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

function make_test_simulation_composite_1d(
    dt::Real = 0.01,
    dkx::Real = 1.0,
    kxmax::Real = 175)

    vf     = u"4.3e5m/s"
    freq   = u"5THz"
    m      = u"20.0meV"
    emax   = u"0.1MV/cm"
    tcycle = uconvert(u"fs", 1 / freq) # 100 fs
    t2     = tcycle / 4             # 25 fs
    t1     = Inf * u"1s"
    σ      = u"800.0fs"

    us      = scaledriving_frequency(freq, vf)
    h       = GappedDirac(energyscaled(m, us))
    l       = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df1     = GaussianAPulse(us, σ, freq, emax)
    df2     = GaussianAPulse(us, σ, 2.5freq, emax)
    df      = df1 + 0.8df2
    tgrid   = SymmetricTimeGrid(dt, -5df1.σ)
    kgrid   = CartesianKGrid1d(dkx, kxmax)
    grid    = NGrid(kgrid, tgrid)
    obs     = [Velocity(grid), Occupation(grid), VelocityX(grid)]

    id    = "sim1d_composite"

    return Simulation(l, df, grid, obs, us, id)
end

function test_composite_1d(sim::Simulation,fns,solver::DamysosSolver;
	atol = 1e-10,
	rtol = 1e-2)
    
    run!(sim, fns, solver; saveplots = true, savedata = true, 
        savepath = joinpath("testresults",sim.id))
	return true
end

function test_load_composite_1d(sim::Simulation, path::String; atol = 1e-10, rtol = 1e-8)
    loaded_sim = Simulation(path)
    return isapprox(sim, loaded_sim, atol=atol, rtol=rtol)    
end

const sim_composite_1d = make_test_simulation_composite_1d()

linchunked = LinearChunked()
const fns_1d_linchunked = define_functions(sim_composite_1d, linchunked)

skipcuda = !(CUDA.functional())

skipcuda &&  @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."

lincuda = skipcuda ? nothing : LinearCUDA(10_000,GPUVern7(),1)
const fns_1d_lincuda = skipcuda ? nothing : define_functions(sim_composite_1d, lincuda)


@testset "CompositeField" begin
    @testset "LinearChunked" begin
        @test test_composite_1d(sim_composite_1d,fns_1d_linchunked,linchunked)
    end
    @testset "LinearCUDA" begin
        @test test_composite_1d(sim_composite_1d,fns_1d_lincuda,lincuda) skip = skipcuda
    end
    @testset "Loading from .hdf5" begin
        @test test_load_composite_1d(
            sim_composite_1d, 
            joinpath("testresults",sim_composite_1d.id,"data.hdf5"))
    end
end
