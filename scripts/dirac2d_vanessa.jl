using Damysos,Unitful,LoggingExtras,Dates

import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"100.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = Inf*tcycle
const σ         = u"100.0fs"

# converged at
# dt = 2.0
# dkx = 0.005

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(0.05,0.1,5,0.5,2.0,-5df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ensid   = "vanessa_100meV"
ens     = parametersweep(sim,sim.numericalparams,
                :dky,
                LinRange(0.05,0.001,8),id=ensid)
ensurepath(sim.plotpath)
logger  = FileLogger(joinpath(ens.plotpath,"dirac2d_vanessa_dky_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
