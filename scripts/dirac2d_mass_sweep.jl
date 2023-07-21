using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"100.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"100.0fs"

dt      = 0.5
dkx     = 0.05 * 0.004
kxmax   = 4.0
dky     = 0.2 * 0.01
kymax   = 1.5

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
obs     = [Velocity(h)]
id      = sprintf1("%x",hash([h,df,pars,obs,us]))
name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
dpath   = "/home/how09898/phd/data/hhgjl/dirac2d_mass_sweep_100fs_8THz_0.3MVcm/"*name
ppath   = "/home/how09898/phd/plots/hhgjl/dirac2d_mass_sweep_100fs_8THz_0.3MVcm/"*name
sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
ensurepath(sim.plotpath)
logger  = FileLogger(joinpath(sim.plotpath,getshortname(sim)*"_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(sim;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(sim::Simulation;...)"
@debug rest
@info "$(now()): calculation finished."
