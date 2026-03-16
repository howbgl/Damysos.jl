using Damysos

if !@isdefined(make_test_simulation_1d)
    function make_test_simulation_1d(
        dt::Real = 0.01,
        dkx::Real = 1.0,
        kxmax::Real = 175;
        id::AbstractString = "sim1d")

        vf = u"4.3e5m/s"
        freq = u"5THz"
        m = u"20.0meV"
        emax = u"0.1MV/cm"
        tcycle = uconvert(u"fs", 1 / freq)
        t2 = tcycle / 4
        σ = u"800.0fs"

        us = scaledriving_frequency(freq, vf)
        h = GappedDirac(energyscaled(m, us))
        l = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
        df = GaussianAPulse(us, σ, freq, emax)
        tgrid = SymmetricTimeGrid(dt, -5df.σ)
        kgrid = CartesianKGrid1d(dkx, kxmax)
        grid = NGrid(kgrid, tgrid)
        obs = [Velocity(grid), Occupation(grid), VelocityX(grid)]

        return Simulation(l, df, grid, obs, us, id)
    end
end

if !@isdefined(make_test_simulation_2d)
    function make_test_simulation_2d(
        dt::Real = 0.01,
        dkx::Real = 1.0,
        dky::Real = 1.0,
        kxmax::Real = 175,
        kymax::Real = 100;
        id::AbstractString = "sim2d")

        vf = u"4.3e5m/s"
        freq = u"5THz"
        m = u"20.0meV"
        emax = u"0.1MV/cm"
        tcycle = uconvert(u"fs", 1 / freq)
        t2 = tcycle / 4
        σ = u"800.0fs"

        us = scaledriving_frequency(freq, vf)
        h = GappedDirac(energyscaled(m, us))
        l = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
        df = GaussianAPulse(us, σ, freq, emax)
        tgrid = SymmetricTimeGrid(dt, -5df.σ)
        kgrid = CartesianKGrid2d(dkx, kxmax, dky, kymax)
        grid = NGrid(kgrid, tgrid)
        obs = [Velocity(grid), Occupation(grid)]

        return Simulation(l, df, grid, obs, us, id)
    end
end

if !@isdefined(make_test_simulation_snap)
    function make_test_simulation_snap(
        dt::Real = 0.01,
        dkx::Real = 2.0,
        dky::Real = 1.0,
        kxmax::Real = 175,
        kymax::Real = 10;
        id::AbstractString = "snapshot_sim")

        vf = u"4.3e5m/s"
        freq = u"5THz"
        m = u"20.0meV"
        emax = u"0.1MV/cm"
        tcycle = uconvert(u"fs", 1 / freq)
        t2 = tcycle / 4
        σ = u"800.0fs"

        us = scaledriving_frequency(freq, vf)
        h = GappedDirac(energyscaled(m, us))
        l = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
        df = GaussianAPulse(us, σ, freq, emax)
        tgrid = SymmetricTimeGrid(dt, -5df.σ)
        kgrid = CartesianKGrid2d(dkx, kxmax, dky, kymax)
        grid = NGrid(kgrid, tgrid)
        ts = collect(Damysos.gettsamples(tgrid))
        obs = [DensityMatrixSnapshots(l, grid; tsamples = ts[1:3:end]), Occupation(grid)]

        return Simulation(l, df, grid, obs, us, id)
    end
end

if !@isdefined(make_test_simulation_composite_1d)
    function make_test_simulation_composite_1d(
        dt::Real = 0.01,
        dkx::Real = 1.0,
        kxmax::Real = 175;
        id::AbstractString = "sim1d_composite")

        vf = u"4.3e5m/s"
        freq = u"5THz"
        m = u"20.0meV"
        emax = u"0.1MV/cm"
        tcycle = uconvert(u"fs", 1 / freq)
        t2 = tcycle / 4
        σ = u"800.0fs"

        us = scaledriving_frequency(freq, vf)
        h = GappedDirac(energyscaled(m, us))
        l = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
        df1 = GaussianAPulse(us, σ, freq, emax)
        df2 = GaussianAPulse(us, σ, 2.5freq, emax)
        df = df1 + 0.8df2
        tgrid = SymmetricTimeGrid(dt, -5df1.σ)
        kgrid = CartesianKGrid1d(dkx, kxmax)
        grid = NGrid(kgrid, tgrid)
        obs = [Velocity(grid), Occupation(grid), VelocityX(grid)]

        return Simulation(l, df, grid, obs, us, id)
    end
end

if !@isdefined(make_test_simulation_tiny)
    function make_test_simulation_tiny(
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
        obs  = [Velocity(grid), Occupation(grid)]

        id    = "sim1"

        return Simulation(l, df, grid, obs, us, id)
    end
end