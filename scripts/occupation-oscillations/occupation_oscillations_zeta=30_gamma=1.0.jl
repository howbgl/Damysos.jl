using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers

import Damysos.getshortname

function make_teelogger(logging_path::AbstractString,name::AbstractString)

      ensurepath(logging_path)
      info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
      info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
      all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

      return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end

function make_system(
      subpath::AbstractString;
      plotpath_base="/home/how09898/phd/plots",
      datapath_base="/home/how09898/phd/data")

      vf        = u"4.3e5m/s"
      m         = u"10.0meV"
      e         = uconvert(u"C",1u"eV"/1u"V")

      ζ         = 30.0
      γ         = 1.0

      M         = ζ * γ
      ω         = 2m / (M * Unitful.ħ)
      freq      = uconvert(u"THz",ω/2π)
      emax      = uconvert(u"MV/cm",ω*m / (vf * e * γ))

      tcycle    = uconvert(u"fs",1/freq)
      t2        = tcycle / 4
      t1        = Inf*u"1s"
      σ         = 2*tcycle

      # for γ=1.0; T2=T/4; T1=∞ converged @
      # dt = 0.01
      # dkx = 0.1
      # dky = 1.0
      # kxmax = 500
      # kymax = 

      # for γ=0.2; T2 = T1 = ∞ converged @
      # dt = 0.01
      # dkx = 0.1
      # dky = 1.0
      # kxmax = 
      # kymax = 

      dt      = 0.01
      dkx     = 0.1
      dky     = 1.0
      kxmax   = 200.0
      kymax   = 10.0

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(us,m,vf,t1,t2)
      df      = GaussianPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "zeta=$(ζ)_gamma=$(γ)"
      name    = "Simulation{$(typeof(h.Δ))}(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

const sim     = make_system("hhgjl/occupation_oscillations/zeta=30/gamma=1.0")
const γ2      = 1.0 / sim.hamiltonian.t2  
const γ2range = LinRange(0.0,γ2,10)
const ens     = parametersweep(sim,sim.numericalparams,:dt,LinRange(0.01,0.001,8))

ensurepath(ens.plotpath)
global_logger(make_teelogger(ens.plotpath,ens.id))

@info "Logging to \"$(ens.plotpath)\""

const results,time,rest... = @timed run_simulation!(ens;
      kxbatch_basesize=256,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
