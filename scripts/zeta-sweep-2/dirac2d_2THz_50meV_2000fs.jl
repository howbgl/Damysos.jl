using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"2THz"
const m         = u"50.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 100 fs
const t2        = tcycle / 4             # 25 fs
const t1        = Inf*u"1s"
const σ         = u"2000.0fs"

# converged at
# dt = 
# dkx = 
# kxmax = 
# dky = 
# kymax = 

const dt      = 0.001
const dkx     = 0.1
const kxmax   = 400.0
const dky     = 1.0
const kymax   = 100.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/zeta-sweep-2/dirac2d_2THz_50meV_2000fs/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/zeta-sweep-2/dirac2d_2THz_50meV_2000fs/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,:dkx,LinRange(0.1,0.01,10))

ensurepath(ens.plotpath)
const info_filelogger  = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now())_debug.log"))
const tee_logger       = TeeLogger(info_logger,all_filelogger)

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log")) " *
      "and $(joinpath(ens.plotpath,getshortname(ens)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
