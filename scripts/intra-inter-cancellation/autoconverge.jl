using Distributed
@everywhere using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers,Dagger

@everywhere import Damysos.getshortname

@everywhere function make_teelogger(logging_path::AbstractString,name::AbstractString)

      ensurepath(logging_path)
      info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
      info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
      all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

      return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end

@everywhere function make_system(
      ζ::Real,
      γ::Real,
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
      
      M         = ζ * γ
      ω         = 2m / (M * Unitful.ħ)
      freq      = uconvert(u"THz",ω/2π)
      σ         = uconvert(u"fs",1/freq)
      emax      = uconvert(u"MV/cm",ω*m / (vf * e * γ))
      us        = scaledriving_frequency(freq,vf)
      df        = GaussianEPulse(us,σ,freq,emax)
      h         = GappedDirac(us,m,vf,t1,t2)

      dt      = 0.01
      ts      = -5df.σ:dt:5df.σ
      kxmax   = 6*maximum_kdisplacement(df,ts)[1]
      kymax   = 1.0
      dkx     = 2kxmax / 1_200
      dky     = 1.0

      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ,rtol,atol)
      obs     = [Velocity(h),Occupation(h)]

      id      = "gamma=$(round(γ,sigdigits=3))_zeta=$(round(ζ,sigdigits=3))"
      name    = "Simulation(2d)_$(id)_$(random_word())"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

@everywhere function make_n_runtest(s)
  method      = SequentialTest(
      [PowerLawTest(:dt,0.5),
      PowerLawTest(:dkx,0.7),
      LinearTest(:kxmax,50.)])
      test        = ConvergenceTest(s,method,1e-12,1e-4)

      run!(test,100,2*60*60,savealldata=false)
end

@everywhere function runall(sims)
      results = []
      @sync for s in sims
            res = Dagger.@spawn make_n_runtest(s)
            push!(results,res)
      end
      return fetch.(results)
end

const keldyshs = LinRange(0.1,2.0,8)
const zetas    = LinRange(1.0,5.0,8)
const sims     = [make_system(
      z,
      g,
      "hhgjl/inter-intra-cancellation/";
      rtol=1e-6,
      atol=1e-12) for g in keldyshs for z in zetas]
const logpath  = dirname(sims[1].datapath)
ensurepath(logpath)
global_logger(make_teelogger(logpath,"all-convergence-tests.log"))
@info "Logging to \"$logpath\"/all-convergence-tests.log"


results     = runall(sims)
str         = repr.("text/plain",results)
for (res,pars) in zip(results,[(z,g) for g in keldyshs for z in zetas])
      @info """
      Result for ζ=$(pars[1]) γ=$(pars[2])

      $res


      """
end

@info "Plotting for last sims"
for r in results
      plotdata(r.test.completedsims[end])
end

