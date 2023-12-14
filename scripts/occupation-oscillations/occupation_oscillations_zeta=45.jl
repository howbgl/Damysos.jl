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

      ζ         = 45.0
      γ         = 0.1

      M         = ζ * γ
      ω         = 2m / (M * Unitful.ħ)
      freq      = uconvert(u"THz",ω/2π)
      emax      = uconvert(u"MV/cm",ω*m / (vf * e * γ))

      tcycle    = uconvert(u"fs",1/freq)
      t2        = tcycle / 4
      t1        = Inf*u"1s"
      σ         = 2*tcycle

      # for T2 = tcycle/4 converged @
      # dt = 0.007
      # dkx = 1.0
      # dky = 1.0
      # kxmax = 500
      # kymax = 300
      dt      = 0.007
      dkx     = 1.0
      dky     = 1.0
      kxmax   = 500.0
      kymax   = 100.0

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

const sim     = make_system("hhgjl/occupation_oscillations/zeta=45")
# const γ2      = 1.0 / sim.hamiltonian.t2  
# const γ2range = LinRange(γ2,10γ2,10)
const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(100.0,400.0,4))

ensurepath(ens.plotpath)
global_logger(make_teelogger(ens.plotpath,ens.id))

@info "Logging to \"$(ens.plotpath)\""

const results,time,rest... = @timed run_simulation!(ens;
      kxbatch_basesize=128,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
