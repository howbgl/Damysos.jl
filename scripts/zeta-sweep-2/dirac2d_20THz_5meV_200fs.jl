using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"20THz"
const m         = u"5.0meV"
const emax      = u"1.0MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 100 fs
const t2        = tcycle / 4             # 25 fs
const t1        = Inf*u"1s"
const σ         = u"200.0fs"

# converged at
# dt = 0.01
# dkx = 0.1
# dky = 0.1
# kxmax = 150 (slight noise variation above 20th harmonic)
# kymax = 50

const dt      = 0.01
const dkx     = 0.1
const kxmax   = 100.0
const dky     = 1.0
const kymax   = 10.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

const id      = "converged"
# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/zeta-sweep-2/dirac2d_20THz_5meV_200fs/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/zeta-sweep-2/dirac2d_20THz_5meV_200fs/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(50.0,50.0,1))

ensurepath(ens.plotpath)
const info_filelogger  = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(ens.plotpath,ens.id*"$(now())_debug.log"))
const tee_logger       = TeeLogger(info_logger,all_filelogger)

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log")) " *
      "and $(joinpath(ens.plotpath,getshortname(ens)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(sim;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
