using Damysos,Unitful,LoggingExtras,Dates

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"100.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"500.0fs"

# converged at
# dt = 2.0
# dky = 0.01
# dkx = 0.005
# kxmax = 4.0
# kymax = 1.0

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(0.005,0.01,4,0.1,2.0,-5df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ens     = parametersweep(sim,sim.numericalparams,
                [:rtol,:atol],
                [[1e-8,1e-8],[1e-11,1e-11],[1e-15,1e-15]])
logger  = FileLogger(joinpath("logs","dirac2d_100meV_tol_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kyparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
