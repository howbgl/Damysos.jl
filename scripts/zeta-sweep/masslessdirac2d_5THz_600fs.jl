using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"5THz"
const m         = u"1e-12meV"             # ≈ 0 meV
const emax      = u"1.0MV/cm"
const tcycle    = uconvert(u"fs",1/freq)  # 200fs
const t1        = Inf*u"1s"
const t2        = u"100.0fs"               # Tcycle/2
const σ         = u"600.0fs"              # 3 Tcycle

# converged at
# dt = 0.001
# dkx = 0.5
# kxmax = 1500
# dky = 
# kymax = 

const dt      = 0.001
const dkx     = 0.5
const kxmax   = 1500.0
const dky     = 100.0
const kymax   = 500.0

us      = scaledriving_frequency(freq,vf)
h       = GappedDirac(us,m,vf,t1,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
obs     = [Velocity(h)]

id      = sprintf1("%x",hash([h,df,pars,obs,us]))
name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
dpath   = "/home/how09898/phd/data/hhgjl/masslessdirac2d_5THz_600fs/"*name
ppath   = "/home/how09898/phd/plots/hhgjl/masslessdirac2d_5THz_600fs/"*name

sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
ens     = parametersweep(sim,sim.numericalparams,
                :dky,
                LinRange(10.0,1.0,10))

ensurepath(ens.plotpath)
logger  = FileLogger(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))"

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
