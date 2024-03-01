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
      ζ::Real,
      γ::Real,
      subpath::AbstractString;
      plotpath_base="/home/how09898/phd/plots",
      datapath_base="/home/how09898/phd/data")

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

      pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
      obs     = [Velocity(h),Occupation(h)]

      id      = "gamma=$(round(γ,sigdigits=3))_zeta=$(round(ζ,sigdigits=3))"
      name    = "Simulation(2d)_$(id)_$(random_word())"
      dpath   = joinpath(datapath_base,subpath,name)
      ppath   = joinpath(plotpath_base,subpath,name)

      return Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
end

const keldyshs = LinRange(0.1,2.0,5)
const sims     = [make_system(g,0.1,"hhgjl/inter-intra-cancellation/") for g in keldyshs]

for s in sims
      method = SequentialTest(PowerLawTest(:dt,0.5),PowerLawTest(:dkx,0.7))
      test = ConvergenceTest(s,method,1e-12,1e-10)
      run!(s,20,60*60)
end

# const γ2cyc   = getparams(sim).ν
# const γ2range = LinRange(1e-4γ2cyc,1e-1γ2cyc,8)
# const ens     = parametersweep(sim,sim.numericalparams,:kymax,LinRange(1,0.1,10))

# ensurepath(ens.plotpath)
# global_logger(make_teelogger(ens.plotpath,sim.id))

# @info "Logging to \"$(ens.plotpath)\""

# const results,time,rest... = @timed run_simulation!(ens;
#       threaded=false,
#       kxbatch_basesize=256,
#       maxparallel_ky=128)

# @info "$(time/60.)min spent in run_simulation!"
# @debug rest
# @info "$(now()): calculation finished."
