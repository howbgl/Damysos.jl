using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"5THz"
const m         = u"20.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = Inf*u"1s"           
const t1        = Inf*u"1s"
const σ         = u"800.0fs"

# For t2 = ∞ converged @
# dt = 0.005
# dkx = 0.05
# dky = 0.4
# kxmax = 200
# kymax = 


# For t2 = Tc / 4 converged @
# dt = 0.01
# dkx = 0.1
# dky = 1.0
# kxmax = 180
# kymax = 50

const dt      = 0.005
const dkx     = 0.05
const dky     = 0.4
const kxmax   = 200.0
const kymax   = 2.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = "converged"
const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/t2-sweep/dirac2d_5THz_20meV_800fs/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/t2-sweep/dirac2d_5THz_20meV_800fs/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(50.0,100.0,5))

ensurepath(ens.plotpath)
global_logger(TerminalLogger())
const info_filelogger  = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(ens.plotpath,ens.id*"$(now())_debug.log"))
const tee_logger       = TeeLogger(global_logger(),info_logger,all_filelogger)

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log")) " *
      "and $(joinpath(ens.plotpath,getshortname(ens)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(ens;
      kxbatch_basesize=512,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!(...)"
@debug rest
@info "$(now()): calculation finished."
