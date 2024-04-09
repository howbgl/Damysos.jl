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

      vf        = u"497070.0m/s"
      m         = u"0.02eV"
      freq      = u"25.0THz"
      emax      = u"0.5MV/cm"
      t2        = Inf*u"1.0s"
      t1        = Inf*u"1.0s"
      σ         = u"40.0fs"

      # for T2 = T1 = ∞ converged @
      # dt = 0.01
      # dkx = 0.1
      # dky = 0.2
      # kxmax = 80
      # kymax = 50

      dt      = 0.001
      dkx     = 0.1
      dky     = 0.1
      kxmax   = 80
      kymax   = 50

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(us,m,vf,t1,t2)
      df      = GaussianEPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "find-coarsest-grid"
      name    = "Simulation(2d)_$(id)_$(random_word())"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

const sim     = make_system("hhgjl/inter-intra-cancellation/")
const γ2cyc   = getparams(sim).ν
const γ2range = LinRange(1e-4γ2cyc,1e-1γ2cyc,8)
const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(1,0.1,10))

ensurepath(ens.plotpath)
global_logger(make_teelogger(ens.plotpath,sim.id))

@info "Logging to \"$(ens.plotpath)\""

const results,time,rest... = @timed run_simulation!(ens;
      threaded=false,
      kxbatch_basesize=256,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!"
@debug rest
@info "$(now()): calculation finished."
