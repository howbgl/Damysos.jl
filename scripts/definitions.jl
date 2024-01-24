using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers,ProgressLogging
using DifferentialEquations,StaticArrays

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"5THz"
const m         = u"20.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 100 fs
const t2        = tcycle / 4             # 25 fs
const t1        = Inf*u"1s"
const σ         = u"300.0fs"


const dt      = 0.01
const dkx     = 1.0
const kxmax   = 256.0
const dky     = 1.0
const kymax   = 4.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const id      = "ref_small"
const name    = "Simulation{$(typeof(h.Δ))}(2d)reference_small"
const dpath   = "test/reference_small"
const ppath   = dpath
const tpath   = "/temp_local/how09898/"

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
