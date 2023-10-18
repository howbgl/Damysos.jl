using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

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
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const id      = "ref"
const name    = "Simulation{$(typeof(h.Δ))}(2d)reference"
const dpath   = "test/reference"
const ppath   = dpath

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)

ensurepath(sim.plotpath)
const info_filelogger  = FileLogger(joinpath(sim.plotpath,sim.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(sim.plotpath,sim.id*"_$(now())_debug.log"))
const console_logger   = ConsoleLogger(stdout)
const tee_logger       = TeeLogger(info_logger,all_filelogger,console_logger)

@info "Logging to $(joinpath(sim.plotpath,getshortname(sim)*"s(now()).log")) " *
      "and $(joinpath(sim.plotpath,getshortname(sim)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(sim;
      threaded=false,
      kxbatch_basesize=64,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!(...)"
@debug rest
@info "$(now()): calculation finished."
