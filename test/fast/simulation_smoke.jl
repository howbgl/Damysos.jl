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
    psim = PreparedSimulation(sim, solver)

    @test fns isa Damysos.SimulationFunctions
    @test Damysos.rhs(fns) isa Tuple
    @test Damysos.bzmask(fns) isa Function
    @test Damysos.observable_functions(fns) isa Vector
    @test psim isa Damysos.PreparedSimulation
    @test psim.sim === sim
    @test psim.solver === solver
    @test psim.functions isa Damysos.SimulationFunctions

    sim_prepared = make_smoke_simulation()
    psim_prepared = PreparedSimulation(sim_prepared, solver)
    res_prepared = run!(
        psim_prepared;
        savedata = false,
        saveplots = false,
        showinfo = false)

    v_prepared = filter(o -> o isa Velocity, res_prepared)[1]
    @test !isempty(v_prepared.vx)
    @test all(isfinite, v_prepared.vx)

    sim_legacy = make_smoke_simulation()
    fns_legacy = define_functions(sim_legacy, solver)
    res_legacy = run!(
        sim_legacy,
        fns_legacy,
        solver;
        savedata = false,
        saveplots = false,
        showinfo = false)

    v_legacy = filter(o -> o isa Velocity, res_legacy)[1]
    @test !isempty(v_legacy.vx)
    @test all(isfinite, v_legacy.vx)
    @test v_prepared.vx ≈ v_legacy.vx
end
