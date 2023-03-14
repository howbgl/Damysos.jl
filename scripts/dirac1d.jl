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
pars    = NumericalParams1d(0.01,10,0.1,-5*df.σ)
obs     = [Velocity(0.0)]
sim     = Simulation(h,df,pars,obs,us,1)
ens     = parametersweep(sim,sim.hamiltonian,:Δ,LinRange(h.Δ,10*h.Δ,4))
run_simulation(ens)

