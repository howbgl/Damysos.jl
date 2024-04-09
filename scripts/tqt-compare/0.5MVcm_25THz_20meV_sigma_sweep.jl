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
      # t2        = Inf*u"1.0s"
      t2        = Inf / freq
      t1        = Inf*u"1.0s"
      σ         = u"40.0fs"

      dt      = 0.001
      dkx     = 0.1
      dky     = 0.1
      # choose such that BZ is 2*0.13Å^-1 x 2*0.13Å^-1
      kxmax   = 38.34750309065676
      kymax   = 25.84764 

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(us,m,vf,t1,t2)
      df      = GaussianEPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "0.5MVcm_25THz_20meV_sigma_sweep"
      name    = "Simulation(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

const sim     = make_system("hhgjl/tqt-compare/")
const kxmax   = getparams(sim).kxmax
const kymax   = getparams(sim).kymax
const ens     = parametersweep(
      sim,
      sim.drivingfield,
      :σ,
      LinRange(1.0,3.0,12),
      id="sigma_sweep_smallBZ_t2=Inf")

ensurepath(ens.plotpath)
global_logger(make_teelogger(ens.plotpath,ens.id))

@info "Logging to \"$(ens.plotpath)\""

const results,time,rest... = @timed run_simulation!(ens;
      threaded=false,
      kxbatch_basesize=256,
      maxparallel_ky=128)

@info "$(time/60.)min spent in run_simulation!"
@debug rest
@info "$(now()): calculation finished."
