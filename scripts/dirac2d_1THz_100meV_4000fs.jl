using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"1THz"
const m         = u"100meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 1000 fs
const t2        = tcycle / 4             # 250 fs
const t1        = Inf*u"1s"
const σ         = u"4000.0fs"

# converged at
# dt = 
# dkx = 
# kxmax = 
# dky = 
# kymax = 

const dt      = 0.01
const dkx     = 0.5
const kxmax   = 100.0
const dky     = 5.0
const kymax   = 100.0

us      = scaledriving_frequency(freq,vf)
h       = GappedDirac(us,m,vf,t1,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
obs     = [Velocity(h)]

id      = sprintf1("%x",hash([h,df,pars,obs,us]))
name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
dpath   = "/home/how09898/phd/data/hhgjl/dirac2d_10THz_10meV_400fs/"*name
ppath   = "/home/how09898/phd/plots/hhgjl/dirac2d_10THz_10meV_400fs/"*name

sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
ens     = parametersweep(sim,sim.numericalparams,
                :kxmax,
                LinRange(0.01,0.005,6))

ensurepath(ens.plotpath)
logger  = FileLogger(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))

@info "Logging to $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))"

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
