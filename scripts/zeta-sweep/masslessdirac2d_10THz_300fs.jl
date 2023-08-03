using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"10THz"
const m         = u"1e-12meV"             # ≈ 0 meV
const emax      = u"1.0MV/cm"
const tcycle    = uconvert(u"fs",1/freq)  # 100fs
const t1        = Inf*u"1s"
const t2        = u"50.0fs"               # Tcycle/2
const σ         = u"300.0fs"              # 3 Tcycle

# converged at
# dt = 0.005
# dkx = 0.1
# dky = 1.0
# kxmax = 375.0
# kymax = 100.0

const dt      = 0.005
const dkx     = 0.1
const dky     = 1.0
const kxmax   = 375.0
const kymax   = 100.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/masslessdirac2d_10THz_300fs/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/masslessdirac2d_10THz_300fs/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,
                :dt,
                LinRange(0.005,0.003,3))

ensurepath(ens.plotpath)
const logger  = FileLogger(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))"

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
