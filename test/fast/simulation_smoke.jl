using Damysos
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))

function make_smoke_simulation()
    vf     = u"4.3e5m/s"
    freq   = u"5THz"
    m      = u"20.0meV"
    emax   = u"0.01MV/cm"
    tcycle = uconvert(u"fs", 1 / freq)
    t2     = tcycle / 4
    σ      = u"200.0fs"

    us    = scaledriving_frequency(freq, vf)
    h     = GappedDirac(energyscaled(m, us))
    l     = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df    = GaussianAPulse(us, σ, freq, emax)
    tgrid = SymmetricTimeGrid(0.1, -2df.σ)
    kgrid = CartesianKGrid1d(5.0, 5.0)
    grid  = NGrid(kgrid, tgrid)
    obs   = [Velocity(grid), Occupation(grid)]

    return Simulation(l, df, grid, obs, us, "sim_smoke")
end

@testset "Simulation Smoke" begin
    sim = make_smoke_simulation()
    solver = LinearChunked(64)
    fns = define_functions(sim, solver)
    res = run!(sim, fns, solver; savedata = false, saveplots = false)

    v = filter(o -> o isa Velocity, res)[1]
    @test !isempty(v.vx)
    @test all(isfinite, v.vx)
end
