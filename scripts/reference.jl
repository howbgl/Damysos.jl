using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers

import Damysos.getshortname
import Damysos.ensurepath


function make_teelogger(logging_path::AbstractString,name::AbstractString)

      ensurepath(logging_path)
      info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
      info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
      all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

      return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end

function make_system(
      subpath::AbstractString;
      plotpath_base="/home/how09898/plots",
      datapath_base="/home/how09898/data")

      vf        = u"4.3e5m/s"
      freq      = u"5THz"
      m         = u"20.0meV"
      emax      = u"0.1MV/cm"
      tcycle    = uconvert(u"fs",1/freq) # 100 fs
      t2        = tcycle / 4             # 25 fs
      t1        = Inf*u"1s"
      σ         = u"800.0fs"

      # converged at
      # dt = 0.01
      # dkx = 1.0
      # dky = 1.0
      # kxmax = 175
      # kymax = 100

      dt      = 0.01
      dkx     = 1.0
      dky     = 1.0
      kxmax   = 175.0
      kymax   = 100.0

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(us,m,vf,t1,t2)
      df      = GaussianPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "ref"
      name    = "Simulation{$(typeof(h.Δ))}(2d)reference"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end


const sim = make_system("hhgjl/reference")
ensurepath(sim.plotpath)
global_logger(make_teelogger(sim.plotpath,sim.id))

@info "Logging to $(joinpath(sim.plotpath,sim.id))\n$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run_simulation!(sim;
      threaded=false,
      savedata=false,
      saveplots=false,
      kxbatch_basesize=64,
      maxparallel_ky=48)

@info "$(time/60.)min spent in run_simulation!(...)"
@debug rest
@info "$(now()): calculation finished."
