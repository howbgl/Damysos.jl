using Damysos,Unitful,LoggingExtras,Dates

const vf        = u"4.3e5m/s"
const freq      = u"12THz"
const m         = u"40.0meV"
const emax      = u"0.5MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"200.0fs"

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(0.1,0.1,5,2,0.2,-5df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ens     = parametersweep(sim,sim.numericalparams,:dt,[1.0,0.5,0.2,0.1])
logger  = FileLogger(joinpath("logs","dirac2d_dt_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kyparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
