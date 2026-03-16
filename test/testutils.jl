using Damysos
using FiniteDifferences
using LoggingExtras
using Random
using TerminalLoggers

if !@isdefined(_DAMYSOS_TEST_LOGGER_SET)
    global_logger(TerminalLogger(stderr, Logging.Info))
    const _DAMYSOS_TEST_LOGGER_SET = true
end

if !@isdefined(_DAMYSOS_TEST_RNG_SEEDED)
    Random.seed!(1234)
    const _DAMYSOS_TEST_RNG_SEEDED = true
end

if !@isdefined(_DAMYSOS_TESTRESULTS_DIR)
    const _DAMYSOS_TESTRESULTS_DIR = begin
        if haskey(ENV, "DAMYSOS_TESTRESULTS_DIR") && !isempty(ENV["DAMYSOS_TESTRESULTS_DIR"])
            dir = ENV["DAMYSOS_TESTRESULTS_DIR"]
            mkpath(dir)
            dir
        else
            mktempdir()
        end
    end
end

if !@isdefined(testresults_dir)
    testresults_dir() = _DAMYSOS_TESTRESULTS_DIR
end

if !@isdefined(datafile)
    datafile(name::AbstractString) = joinpath(@__DIR__, "data", name)
end

if !@isdefined(test_1d)
    function test_1d(v_ref::Velocity, sim::Simulation, fns, solver::DamysosSolver;
        atol = 1e-10,
        rtol = 1e-2)

        res = run!(sim, fns, solver; saveplots = false, savedata = true, showinfo = false,
            savepath = joinpath(testresults_dir(), Damysos.getname(sim)))
        v = filter(o -> o isa Velocity, res)[1]
        vx = filter(o -> o isa VelocityX, res)[1]
        vx_ref = VelocityX(v_ref.vx, v_ref.vxintra, v_ref.vxinter)
        return isapprox(v, v_ref, atol = atol, rtol = rtol) &&
            isapprox(vx, vx_ref, atol = atol, rtol = rtol)
    end
end

if !@isdefined(test_2d)
    function test_2d(v_ref::Velocity, sim::Simulation, fns, solver::DamysosSolver;
        atol = 1e-10,
        rtol = 1e-2)

        res = run!(sim, fns, solver; saveplots = false, showinfo = false,
            savepath = joinpath(testresults_dir(), Damysos.getname(sim)))
        v = filter(o -> o isa Velocity, res)[1]
        return isapprox(v, v_ref, atol = atol, rtol = rtol)
    end
end

if !@isdefined(test_snapshots)
    function test_snapshots(sim::Simulation, fns, solver; atol = 1e-10, rtol = 1e-8)
        run!(sim, fns, solver; savedata = false, saveplots = false, showinfo = false)

        ks = Damysos.getksamples(sim.grid.kgrid)
        dms = sim.observables[1]
        ts = dms.tsamples

        occ_ref = sim.observables[2]
        cbocc = empty(occ_ref.cbocc)
        obs = [Occupation(cbocc)]
        bzmask = fns[2]

        for (i, dm) in enumerate(dms.density_matrices)
            cc = [real(m[1, 1]) for m in dm.density_matrix]
            weights = bzmask.(ks, ts[i])
            push!(cbocc, sum(cc .* weights))
        end

        Damysos.applyweights_afterintegration!(obs, sim.grid.kgrid)
        Damysos.normalize!.(obs, (2pi)^sim.dimensions)

        full_ts = Damysos.gettsamples(sim.grid)
        cbocc_ref = empty(occ_ref.cbocc)

        for t in ts
            i = Damysos.find_index_nearest(t, full_ts)
            push!(cbocc_ref, occ_ref.cbocc[i])
        end

        return isapprox(cbocc, cbocc_ref, atol = atol, rtol = rtol)
    end
end

if !@isdefined(check_tensor_data)
    function check_tensor_data(data1, data2; atol = 1e-12, rtol = 1e-12)
        ia(a, b) = Base.isapprox(a, b, atol = atol, rtol = rtol)
        isanynan(x) = x isa AbstractArray ? any(isnan.(x)) : isnan(x)

        return all([any(isanynan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
    end
end

if !@isdefined(check_jacobian)
    function check_jacobian(fn, dfn; krange = -1:0.03:1, atol = 1e-12, rtol = 1e-12)
        fd = central_fdm(5, 1)
        data = [dfn(kx, ky) for kx in krange, ky in krange]
        data_fdm = [jacobian(fd, fn, [kx, ky])[1] for kx in krange, ky in krange]
        return check_tensor_data(data, data_fdm; atol = atol, rtol = rtol)
    end
end

if !@isdefined(sample)
    function sample(f, krange)
        return [f(kx, ky) for kx in krange, ky in krange]
    end
end

if !@isdefined(check_scalar)
    function check_scalar(a, b; krange = -1:0.03:1, atol = 1e-12, rtol = 1e-12)
        
        ia(a, b)    = Base.isapprox(a, b, atol = atol, rtol = rtol)
        data1       = sample(a, krange)
        data2       = sample(b, krange)

        return all([any(isnan.([a, b])) ? true : ia(a, b) for (a, b) in zip(data1, data2)])
    end
end

if !@isdefined(check_dhdkm)
    function check_dhdkm(fn, dfn, kindex; krange = -1:0.03:1, atol = 1e-12, rtol = 1e-12)
        fd = central_fdm(5, 1)

        data = sample(dfn, krange)
        data_fdm = sample((kx, ky) -> jacobian(fd, fn, [kx, ky])[1][:, kindex], krange)

        return check_tensor_data(data, data_fdm; atol = atol, rtol = rtol)
    end
end