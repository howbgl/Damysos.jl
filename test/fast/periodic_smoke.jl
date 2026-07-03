using Damysos
using Test

include(joinpath(@__DIR__, "..", "testutils.jl"))

function make_sctoy1d_smoke()
    freq   = u"5THz"
    emax   = u"0.01MV/cm"
    tcycle = uconvert(u"fs", 1 / freq)
    t2     = tcycle / 4
    σ      = u"200.0fs"

    us    = UnitScaling(u"1.0fs", u"1.0Å")
    h     = SemiconductorToy1d(us)
    l     = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df    = GaussianAPulse(us, σ, freq, emax)
    tgrid = SymmetricTimeGrid(0.1, -2df.σ)
    kgrid = CartesianMPKGrid1d(2π / (4h.a), h.a)  # 4 k-points
    grid  = NGrid(kgrid, tgrid)
    obs   = [Velocity(grid)]
    return Simulation(l, df, grid, obs, us, "sctoy1d_smoke")
end

function make_hbn_smoke()
    freq   = u"5THz"
    emax   = u"0.01MV/cm"
    tcycle = uconvert(u"fs", 1 / freq)
    t2     = tcycle / 4
    σ      = u"200.0fs"

    us    = UnitScaling(u"1.0fs", u"1.0Å")
    h     = MonolayerhBN(us)
    l     = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df    = GaussianAPulse(us, σ, freq, emax)
    tgrid = SymmetricTimeGrid(0.1, -2df.σ)
    kgrid = HexagonalMPKGrid2d(π / 5, π / 5, h.a)  # 5×5 = 25 k-points
    grid  = NGrid(kgrid, tgrid)
    obs   = [Velocity(grid)]
    return Simulation(l, df, grid, obs, us, "hbn_smoke")
end

@testset "Periodic k-grid construction" begin
    @testset "CartesianMPKGrid1d" begin
        kg = CartesianMPKGrid1d(π / 10, 1.0)
        @test kg.dkx ≈ 2π / round(Int, 2π / (π / 10))
        @test Damysos.ntrajectories(kg) == round(Int, 2π / (π / 10))
        @test length(Damysos.getksamples(kg)) == Damysos.ntrajectories(kg)
        @test_throws ArgumentError CartesianMPKGrid1d(5.0, 1.0)  # q1 < 2

        # lattice constant a != 1: spacing and sample count derive from BZ length 2π/a
        kg2 = CartesianMPKGrid1d(2π / (5 * 2.0), 2.0)
        @test kg2.dkx ≈ 2π / (5 * 2.0)
        @test Damysos.ntrajectories(kg2) == 5
        @test length(Damysos.getksamples(kg2)) == 5
    end

    @testset "HexagonalMPKGrid2d" begin
        us = UnitScaling(u"1.0fs", u"1.0Å")
        h  = MonolayerhBN(us)
        kg = HexagonalMPKGrid2d(π / 5, π / 5, h.a)
        @test Damysos.ntrajectories(kg) > 0
        @test Damysos.getdimension(kg) == UInt8(2)
        @test_throws ArgumentError HexagonalMPKGrid2d(100.0, 100.0, h.a)  # q < 2

        # integration weight includes the 120° cell Jacobian sin factor
        @test Damysos.volume_element(kg) ≈ (√3 / 2) * kg.dk1 * kg.dk2

        # b1, b2 must be reciprocal translations of the hBN Hamiltonian
        b1, b2 = Damysos.reciprocal_primitive_vectors(kg)
        for b in (b1, b2), k in ([0.3, -0.7], [1.1, 0.4])
            @test Damysos.hvec(h, k...) ≈ Damysos.hvec(h, (k .+ b)...)
        end
    end
end

@testset "Periodic grid HDF5 round-trip" begin
    us = UnitScaling(u"1.0fs", u"1.0Å")
    objs = [
        CartesianMPKGrid1d(2π / (4 * 2.82), 2.82),
        HexagonalMPKGrid2d(π / 5, π / 5, 2.51),
        SemiconductorToy1d(us),
        MonolayerhBN(us)]

    path = joinpath(testresults_dir(), "periodic_roundtrip.hdf5")
    Damysos.HDF5.h5open(path, "w") do file
        for (i, obj) in enumerate(objs)
            Damysos.generic_save_hdf5(obj, file, "obj$i")
        end
    end
    Damysos.HDF5.h5open(path, "r") do file
        for (i, obj) in enumerate(objs)
            @test Damysos.load_obj_hdf5(file["obj$i"]) == obj
        end
    end
end

@testset "Periodic model compatibility check" begin
    us    = UnitScaling(u"1.0fs", u"1.0Å")
    h     = SemiconductorToy1d(us)
    freq  = u"5THz"
    emax  = u"0.01MV/cm"
    t2    = uconvert(u"fs", 1 / freq) / 4
    l     = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df    = GaussianAPulse(us, u"200.0fs", freq, emax)
    tgrid = SymmetricTimeGrid(0.1, -2.0)
    kgrid = CartesianKGrid1d(5.0, 5.0)  # aperiodic grid — incompatible with periodic Liouvillian
    grid  = NGrid(kgrid, tgrid)
    obs   = [Velocity(grid)]
    @test_throws ArgumentError Simulation(l, df, grid, obs, us, "compat_test")
end

@testset "SemiconductorToy1d smoke" begin
    sim    = make_sctoy1d_smoke()
    solver = LinearChunked(16)
    fns    = define_functions(sim, solver)

    @test fns isa Damysos.SimulationFunctions

    res = run!(sim, fns, solver; savedata = false, saveplots = false, showinfo = false)
    v   = filter(o -> o isa Velocity, res)[1]
    @test !isempty(v.vx)
    @test all(isfinite, v.vx)
end

@testset "MonolayerhBN smoke" begin
    sim    = make_hbn_smoke()
    solver = LinearChunked(16)
    fns    = define_functions(sim, solver)

    @test fns isa Damysos.SimulationFunctions

    # showinfo=true path dispatches printBZSI on the hexagonal grid
    @test Damysos.printBZSI(sim.drivingfield, sim.grid.kgrid, sim.unitscaling) isa String

    res = run!(sim, fns, solver; savedata = false, saveplots = false, showinfo = false)
    v   = filter(o -> o isa Velocity, res)[1]
    @test !isempty(v.vx)
    @test all(isfinite, v.vx)
end
