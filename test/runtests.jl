using CSV
using Damysos
using DataFrames
using Test

function make_test_simulation1()

    vf        = u"4.3e5m/s"
    freq      = u"5THz"
    m         = u"20.0meV"
    emax      = u"0.1MV/cm"
    tcycle    = uconvert(u"fs",1/freq) # 100 fs
    t2        = tcycle / 4             # 25 fs
    t1        = Inf*u"1s"
    σ         = u"800.0fs"

    # converged at
    # dt = 0.01
    # dkx = 1.0
    # dky = 1.0
    # kxmax = 175
    # kymax = 100

    dt      = 0.01
    dkx     = 1.0
    kxmax   = 175.0
    dky     = 1.0
    kymax   = 100.0

    us      = scaledriving_frequency(freq,vf)
    h       = GappedDirac(energyscaled(m,us))
    l       = TwoBandDephasingLiouvillian(h,Inf,timescaled(t2,us))
    df      = GaussianAPulse(us,σ,freq,emax)
    pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
    obs     = [Velocity(pars),Occupation(pars)]

    id      = "sim1"
    dpath   = "testresults/sim1"
    ppath   = "testresults/sim1"

    return Simulation(l,df,pars,obs,us,id,dpath,ppath)
end

const sim1 = make_test_simulation1()
const linchunked = LinearChunked()
const fns = define_functions(sim1,linchunked)
const referencedata = DataFrame(CSV.File("referencedata.csv"))
const vref = Velocity(
    referencedata.vx,
    referencedata.vxintra,
    referencedata.vxinter,
    referencedata.vy,
    referencedata.vyintra,
    referencedata.vyinter)

@testset "Damysos.jl" begin
    @testset "Simulation 1" begin
        res = run!(sim1,fns,linchunked)
        v   = sim1.observables[1]
        @test isapprox(v,vref,atol=1e-10,rtol=1e-2)
    end
end
