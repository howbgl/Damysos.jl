using Unitful,Pkg

Pkg.activate(".")

using Damysos

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"10meV"
const emax      = u"0.28MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"500fs"

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams1d(0.01,10,0.1,-5*df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,1)
ens     = parametersweep(sim,sim.drivingfield,:σ,[df.σ,df.σ/2,df.σ/4,df.σ/8])
run_simulation!(ens)

