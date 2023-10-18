using Damysos,Unitful,LoggingExtras,Dates

import Damysos.ensurepath
import Damysos.run_simulation1d_serial!

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"10.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"500.0fs"

const us,h    = scalegapped_dirac(m,vf,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(0.01,0.1,10,10,0.1,-5df.σ)
const obs     = [Velocity(pars)]
const sim     = Simulation(h,df,pars,obs,us,2,"new_kxparallel")
ensurepath(sim.plotpath)

const info_filelogger  = FileLogger(joinpath(sim.plotpath,"kxpartest_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(sim.plotpath,"kxpartest_$(now())_debug.log"))
const tee_logger       = TeeLogger(info_logger,all_filelogger)

global_logger(tee_logger)

@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(sim;kxparallel=true,kx_workers=64)


@info "$(time/60.)min spent in run_simulation!(...)"
@debug rest
@info "$(now()): calculation finished."
