using Damysos,Unitful,LoggingExtras,Dates

import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"10.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"500.0fs"

const us,h    = scalegapped_dirac(m,vf,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(0.1,0.1,10,3,0.1,-5df.σ)
const obs     = [Velocity(h)]
const sim     = Simulation(h,df,pars,obs,us,2)
ensurepath(sim.plotpath)
const logger  = FileLogger(joinpath(sim.plotpath,"kxpartest_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(sim;kxparallel=true)


@info "$(time/60.)min spent in run_simulation!(...)"
@debug rest
@info "$(now()): calculation finished."
