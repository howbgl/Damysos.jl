using Damysos,Unitful,LoggingExtras,Dates,Folds

const vf        = u"4.3e5m/s"
const freq      = u"12THz"
const m         = u"40.0meV"
const emax      = u"0.5MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"200.0fs"

us,h    = scalegapped_dirac(m,vf,t2)
df      = GaussianPulse(us,σ,freq,emax)
pars    = NumericalParams1d(0.01,5,0.1,-5*df.σ)
obs     = [Velocity(h)]
sim     = Simulation(h,df,pars,obs,us,2)
ens     = parametersweep(sim,sim.numericalparams,:dkx,LinRange(0.001,0.0001,64))

kpoints = [getparams(s).nkx for s in ens.simlist]

times   = Folds.collect(@elapsed run_simulation!(s,
            saveplots=false,
            savedata=false) for s in ens.simlist)

