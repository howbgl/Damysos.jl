using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"1.0THz"
const m         = u"100.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1.0/freq) # 1000 fs
const t2        = tcycle / 4.0             # 250 fs
const t1        = Inf*u"1.0s"
const σ         = u"4000.0fs"

# converged at
# dt = 
# dkx = 
# kxmax = 
# dky = 
# kymax = 

const dt      = 0.01
const dkx     = 5.0
const kxmax   = 1000.0
const dky     = 1.0
const kymax   = 400.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/dirac2d_1THz_100meV_4000fs/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/dirac2d_1THz_100meV_4000fs/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,
                :dt,
                LinRange(0.01,0.005,6))

ensurepath(ens.plotpath)
const logger  = FileLogger(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))"

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
