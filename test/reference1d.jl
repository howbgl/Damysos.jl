using CUDA
using CSV
using Damysos
using DataFrames
using LoggingExtras
using TerminalLoggers
using Test

function make_test_simulation_1d(
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

    us   = scaledriving_frequency(freq, vf)
    h    = GappedDirac(energyscaled(m, us))
    l    = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df   = GaussianAPulse(us, σ, freq, emax)
    ky 	 = 0.0
    pars = NumericalParams1d(dkx, kxmax, ky, dt, -5df.σ)
    obs  = [Velocity(pars), Occupation(pars)]

    id    = "sim1d"
    dpath = "testresults/sim1d"
    ppath = "testresults/sim1d"

    return Simulation(l, df, pars, obs, us, id, dpath, ppath)
end

function test_1d(v_ref::Velocity,sim::Simulation,fns,solver::DamysosSolver;
	atol = 1e-10,
	rtol = 1e-2)
    
    res = run!(sim, fns, solver; saveplots = false)
	v   = filter(o -> o isa Velocity,res)[1]
	return isapprox(v, v_ref, atol = atol, rtol = rtol)
end

const referencedata1d = DataFrame(CSV.File("referencedata1d.csv"))
const vref1d = Velocity(
	referencedata1d.vx,
	referencedata1d.vxintra,
	referencedata1d.vxinter,
	referencedata1d.vy,
	referencedata1d.vyintra,
	referencedata1d.vyinter)

const sim_1d = make_test_simulation_1d()
const sim_1d_dkx = make_test_simulation_1d(0.01,2.0)

linchunked = LinearChunked()
const fns_1d_linchunked = define_functions(sim_1d, linchunked)

skipcuda = false

try
	LinearCUDA()
catch err
	if err == ErrorException("CUDA.jl is not functional, cannot use LinearCUDA solver.")
		global skipcuda = true
		@warn "Skipping CUDA tests, CUDA.jl is not functional."
	end
end
lincuda = skipcuda ? nothing : LinearCUDA()
const fns_1d_lincuda = skipcuda ? nothing : define_functions(sim_1d, lincuda)


@testset "Reference (1d)" begin
    @testset "LinearChunked" begin
        @test test_1d(vref1d,sim_1d,fns_1d_linchunked,linchunked)
    end
    @testset "LinearCUDA" begin
        @test test_1d(vref1d,sim_1d,fns_1d_lincuda,lincuda) skip = skipcuda
    end
end