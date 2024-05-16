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
      kxmax=80.0,
      plotpath_base="/home/how09898/phd/plots",
      datapath_base="/home/how09898/phd/data")

      vf        = u"497070.0m/s"
      m         = u"15.0meV"
      freq      = u"25.0THz"
      emax      = u"0.1MV/cm"
      t2        = Inf*u"1.0s"
      t1        = Inf*u"1.0s"
      σ         = u"40.0fs"

      # for T2 = T1 = ∞ converged @
      # dt = 0.01
      # dkx = 0.1
      # dky = 0.1
      # kxmax =
      # kymax =

      dt      = 0.01
      dkx     = 0.1
      dky     = 1.0
      # kxmax   = 80.0
      kymax   = 5.0

      us      = scaledriving_frequency(freq,vf)
      h       = GappedDirac(energyscaled(m,us))
      l       = TwoBandDephasingLiouvillian(h,timescaled(t1,us),timescaled(t2,us))
      df      = GaussianEPulse(us,σ,freq,emax)
      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "1.0MVcm_25THz_15meV"
      name    = "Simulation(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(l,df,pars,obs,us,id,dpath,ppath)
end

for kxmax in LinRange(80.0,160.0,5)
      sim = make_system("hhgjl/tqt-compare-newcode/kxmax=$kxmax",kxmax=kxmax)
      ensurepath(sim.plotpath)
      global_logger(make_teelogger(sim.plotpath,sim.id))

      @info "Logging to \"$(sim.plotpath)\""

      solver = LinearChunked(512)
      functions = define_functions(sim,solver)
      results,time,rest... = @timed run!(sim,functions,solver)

      @info "$(time/60.)min spent in run_simulation!"
      @debug rest
end

@info "$(now()): calculation finished."
