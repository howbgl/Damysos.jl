using Unitful,Pkg

Pkg.activate(".")

using Damysos

const vf        = u"4.3e5m/s"
const freq      = u"12THz"
const m         = u"10meV"
const emax      = u"0.5MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = 2tcycle

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams2d(0.1,0.1,10,0.1,0.1,-5df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(0.1,1,4))
run_simulation!(ens)
