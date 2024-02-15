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

      dt      = 0.001
      dkx     = 0.1
      dky     = 0.1
      # choose such that BZ is 0.13Å^-1 x 0.13Å^-1
      kxmax   = 38.34750309065676
      kymax   = 25.84764 

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(us,m,vf,t1,t2)
      df      = GaussianEPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "0.5MVcm_25THz_20meV_smallBZ"
      name    = "Simulation(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

const sim     = make_system("hhgjl/tqt-compare/")

ensurepath(sim.plotpath)
global_logger(make_teelogger(sim.plotpath,sim.id))

@info "Logging to \"$(sim.plotpath)\""

const results,time,rest... = @timed run_simulation!(sim;
      threaded=false,
      kxbatch_basesize=256,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!"
@debug rest
@info "$(now()): calculation finished."
