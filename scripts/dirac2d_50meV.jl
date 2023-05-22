using Damysos,Unitful,LoggingExtras,Dates

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"50.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"300.0fs"

# converged at
# dt = 0.5
# dkx = 0.08

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(0.08,0.05,4,0.1,0.5,-5df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ens     = parametersweep(sim,sim.numericalparams,:kxmax,[3.0,4.0,5.0,6.0])
logger  = FileLogger(joinpath("logs","dirac2d_50meV_300fs_kxmax_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;ensembleparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
