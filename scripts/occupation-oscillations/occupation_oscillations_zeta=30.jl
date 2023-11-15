using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers

import Damysos.getshortname

const vf        = u"4.3e5m/s"
const m         = u"10.0meV"
const e         = uconvert(u"C",1u"eV"/1u"V")

const ζ         = 30.0
const γ         = 0.1

const M         = ζ * γ
const ω         = 2m / (M * Unitful.ħ)
const freq      = uconvert(u"THz",ω/2π)
const emax      = uconvert(u"MV/cm",ω*m / (vf * e * γ))


const tcycle    = uconvert(u"fs",1/freq)
const t2        = tcycle / 2
const t1        = Inf*u"1s"
const σ         = 2*tcycle

# for T2 = T1 = ∞ converged @
# dt = 0.01 
# dkx = 0.1
# dky = 1.0
# kxmax = 330
# kymax = 150

const dt      = 0.01
const dkx     = 0.1
const dky     = 1.0
const kxmax   = 330.0
const kymax   = 0.1


const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h),Occupation(h)]

const γ2      = 1.0 / h.t2  
const γ2range = LinRange(0.0,γ2,10)

const id      = "zeta=$(ζ)_gamma=$(γ)"
const name    = "Simulation{$(typeof(h.Δ))}(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/occupation_oscillations/zeta=30/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/occupation_oscillations/zeta=30/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(1.0,150,10))

global_logger(TerminalLogger())
ensurepath(ens.plotpath)
const info_filelogger  = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(ens.plotpath,ens.id*"_$(now())_debug.log"))
const tee_logger       = TeeLogger(global_logger(),info_logger,all_filelogger)

@info "Logging to\n    $(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log")) " *
      "and\n    $(joinpath(ens.plotpath,getshortname(ens)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(ens;
      kxbatch_basesize=256,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
