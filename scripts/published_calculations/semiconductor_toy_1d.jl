##########################################################
# Script for reproducing published results
##########################################################
#
# * published in PhD thesis
# * TODO put doi
#
# TODO put figure number & page.
##########################################################

function make_simulation()

    freq    = uconvert(u"THz", Unitful.c0 / u"3.25μm")
    emax    = u"0.15V/Å"
    tcycle  = uconvert(u"fs", 1 / freq) # approx 10.83 fs
    t2      = tcycle / 4             # approx 2.71 fs
    σ       = 4.0 * tcycle  # approx 21.66 fs

    us      = UnitScaling(u"1.0fs", u"1.0Å")
    h       = SemiconductorToy1d(us,u"1.65eV")
    l       = TwoBandDephasingLiouvillian(h, Inf, timescaled(t2, us))
    df      = GaussianAPulse(us, σ, freq, emax)

    dt      = timescaled(tcycle, us) / 1_000
    dk      = 2π / (3_000h.a)
    tgrid   = SymmetricTimeGrid(dt, -5df.σ)
    kgrid   = CartesianMPKGrid1d(dk, h.a)
    grid    = NGrid(kgrid, tgrid)
    obs     = [Velocity(grid)]
    id      = "semiconductortoy1d"

    return Simulation(l, df, grid, obs, us, id)
end

const sim    = make_simulation()
const solver = if CUDA.functional()
    LinearCUDA()
else
    LinearChunked()
end
const fns = define_functions(sim,solver)
const res = run!(sim, fns, solver; savepath="SC_toy1d")