using Unitful,LoggingExtras,Dates,Formatting,DifferentialEquations,TerminalLoggers
using Distributed

@everywhere using Damysos
@everywhere using StaticArrays


import Damysos.getshortname
import Damysos.ensurepath

global_logger(TerminalLogger())

const vf        = u"4.3e5m/s"
const freq      = u"5THz"
const m         = u"20.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 100 fs
const t2        = tcycle / 4             # 25 fs
const t1        = Inf*u"1s"
const σ         = u"800.0fs"

# converged at
# dt = 0.01
# dkx = 1.0
# dky = 1.0
# kxmax = 175
# kymax = 100

const dt      = 0.01
const dkx     = 1.0
const kxmax   = 175.0
const dky     = 1.0
const kymax   = 100.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(energyscaled(m,us))
const l       = TwoBandDephasingLiouvillian(h,Inf,timescaled(t2,us))
const df      = GaussianAPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const id      = "ref"
const name    = "Simulation{$(typeof(h.m))}(2d)reference"
const dpath   = "/home/how09898/phd/data/hhgjl/test/reference"
const ppath   = "/home/how09898/phd/plots/hhgjl/test/reference"

const sim     = Simulation(l,df,pars,obs,us,2,id,dpath,ppath)

@everywhere @eval rhs(u,p,t)        = $(buildrhs_x_expression(sim.liouvillian,sim.drivingfield))
@everywhere @eval bzmask(kx,ky,t)   = $(buildbzmask_expression(sim))
@everywhere @eval f(u,kx,ky,t)      = $(buildobservable_expression(sim))

const prob              = buildensemble_linear(sim,rhs,bzmask,f,reduction)
const ts                = collect(gettsamples(sim.numericalparams))
sol                     = solve(
    prob,
    nothing,
    EnsembleThreads(),
    saveat=ts,
    trajectories=1_000,
    progress=true)
