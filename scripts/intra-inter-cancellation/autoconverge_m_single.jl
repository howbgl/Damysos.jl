using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers,ArgParse

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

      dt      = 0.005
      # ts      = -5df.σ:dt:5df.σ
      ts      = -5df.σ:dt:-df.σ
      kxmax   = 10*maximum_kdisplacement(df,ts)[1]
      kymax   = 1.0
      dkx     = 2kxmax / 1_200
      dky     = 1.0

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
      [PowerLawTest(:dt,0.5),
      PowerLawTest(:dkx,0.5),
      LinearTest(:kxmax,50.)])
      test        = ConvergenceTest(s,method,1e-12,1e-3)

      run!(test,100,4*60*60,savealldata=false)
end

function parse_cmdargs()
      s = ArgParseSettings()
      @add_arg_table! s begin
      "--multiphoton", "-m"
            help = "multiphoton parameter"
            arg_type = Int
      "--zeta", "-z"
            help = "zeta parameter"
            arg_type = Int
      end
      return parse_args(s)
end

const cmdargs  = parse_cmdargs()
const ms       = LinRange(1.0,20.0,8)
const zetas    = LinRange(1.0,5.0,8)
const M        = ms[cmdargs["multiphoton"]]
const z        = zetas[cmdargs["zeta"]]

@info "ζ = $z γ = $M"

const sim = make_system(
      z,
      M,
      "hhgjl/inter-intra-cancellation/multiphoton/longer";
      rtol=1e-4,
      atol=1e-12)

const logpath  = dirname(sim.datapath)

ensurepath(logpath)
global_logger(make_teelogger(logpath,"convergence-tests-z=$z-M=$M.log"))
@info "Logging to \"$logpath\"/convergence-tests-z=$z-M=$M.log"

res         = make_n_runtest(sim)
str         = repr("text/plain",res)
@info """
Result for ζ = $z γ = $M

$str


"""

@info "Plotting for last sim"
plotdata(res.test.completedsims[end])

@info "Finished"