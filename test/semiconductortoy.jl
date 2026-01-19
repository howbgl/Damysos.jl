using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

function make_test_simulation_sc_toy1d()

    freq    = uconvert(u"THz", Unitful.c0 / u"3.25μm") # approx 92.3 THz
    emax    = u"0.15V/Å"
    tcycle  = uconvert(u"fs", 1 / freq) # approx 10.83 fs
    t2      = tcycle / 4             # approx 2.71 fs
    σ       = 2.0 * tcycle  # approx 21.66 fs
    
    us      = UnitScaling(u"1.0fs", u"1.0Å")
    h       = SemiconductorToy1d(us)
    l       = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df      = GaussianAPulse(us, σ, freq, emax)

    dt      = timescaled(tcycle, us) / 1_000
    dk      = 2π / (3_000h.a)
    tgrid   = SymmetricTimeGrid(dt, -5df.σ)
    kgrid   = CartesianMPKGrid1d(dk, h.a)
    grid    = NGrid(kgrid, tgrid)
    obs     = [Velocity(grid)]

    id    = "semiconductortoy1d"

    return Simulation(l, df, grid, obs, us, id)
end

function test_sctoy1d(sim::Simulation,fns,solver::DamysosSolver)
    
    run!(sim, fns, solver; saveplots = true, savedata = true, 
        savepath = joinpath("testresults",Damysos.getname(sim)))
	return true
end


const sim_sctoy1d = make_test_simulation_sc_toy1d()

linchunked = LinearChunked()
const fns_sctoy1d_linchunked = define_functions(sim_sctoy1d, linchunked)

skipcuda = !(CUDA.functional())

skipcuda &&  @warn "Skipping CUDA tests, CUDA.jl is not functional (mark as broken)."
lincuda = skipcuda ? nothing : LinearCUDA(10_000,GPUVern7(),1)
const fns_sctoy1d_lincuda = skipcuda ? nothing : define_functions(sim_sctoy1d, lincuda)


@testset "Reference (1d)" begin
    @testset "LinearChunked" begin
        @test test_sctoy1d(sim_sctoy1d,fns_sctoy1d_linchunked,linchunked)
    end
    @testset "LinearCUDA" begin
        @test test_sctoy1d(sim_sctoy1d,fns_sctoy1d_lincuda,lincuda) skip = skipcuda
    end
end
