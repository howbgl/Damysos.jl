using Damysos,Unitful,LoggingExtras,Dates,TerminalLoggers

import Damysos.getshortname

function make_teelogger(logging_path::AbstractString,name::AbstractString)

      ensurepath(logging_path)
      info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
      info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
      all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

      return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end

function make_system(
      ζ::Real,
      M::Real,
      subpath::AbstractString;
      plotpath_base="/home/how09898/phd/plots",
      datapath_base="/home/how09898/phd/data",
      rtol=1e-12,
      atol=1e-12)

      vf        = u"497070.0m/s"
      m         = u"0.02eV"
      # freq      = u"25.0THz"
      # emax      = u"0.5MV/cm"
      t2        = Inf*u"1.0s"
      t1        = Inf*u"1.0s"
      e         = uconvert(u"C",1u"eV"/1u"V")
      
      γ         = M / ζ
      ω         = 2m / (M * Unitful.ħ)
      freq      = uconvert(u"THz",ω/2π)
      σ         = uconvert(u"fs",1/freq)
      emax      = uconvert(u"MV/cm",ω*m / (vf * e * γ))
      us        = scaledriving_frequency(freq,vf)
      df        = GaussianEPulse(us,σ,freq,emax)
      h         = GappedDirac(us,m,vf,t1,t2)

      dt      = 0.0025
      ts      = -5df.σ:dt:-df.σ
      kxmax   = 10*maximum_kdisplacement(df,ts)[1]
      kymax   = 1.0
      # dkx     = 2kxmax / 1_200
      dkx     = 0.0476
      dky     = 0.0312

      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ,rtol,atol)
      obs     = [Velocity(h),Occupation(h)]

      id      = "M=$(round(M,sigdigits=3))_zeta=$(round(ζ,sigdigits=3))"
      name    = "Simulation(2d)_$id"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

function make_n_runtest(s)
      method      = SequentialTest(
            [LinearTest(:kxmax,50.),
            LinearTest(:kymax,50.)])
      test        = ConvergenceTest(s,method,1e-12,1e-4)

      run!(test,100,14*60*60,savealldata=false,sequentialsim=false)
end


const sim = make_system(3.0,1.0,"hhgjl/inter-intra-cancellation/multiphoton/left";
      rtol=1e-7,atol=1e-12)

const logpath  = dirname(sim.datapath)
ensurepath(logpath)
global_logger(make_teelogger(logpath,"convergence-tests.log"))
@info "Logging to \"$logpath\"/convergence-tests.log"

res         = make_n_runtest(sim)


@info "Plotting for last sims"
plotdata(res.test.completedsims[end])

@info "Finished"

