import Damysos.getshortname

using ArgParse
using Damysos
using Dates
using LoggingExtras
using TerminalLoggers
using Unitful
      
function make_teelogger(
      logging_path::AbstractString,
      name::AbstractString,
      loglevel::LogLevel=Logging.Info)

      ensurepath(logging_path)
      info_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now()).log")
      info_logger = MinLevelLogger(info_filelogger, Logging.Info)
      all_filelogger = FileLogger(joinpath(logging_path, name) * "_$(now())_debug.log")

      return TeeLogger(TerminalLogger(stderr,loglevel), info_logger, all_filelogger)
end

function make_system(
      ζ::Real,
      M::Real,
      subpath::AbstractString;
      plotpath_base="/home/how09898/phd/plots",
      datapath_base="/home/how09898/phd/data",
      rtol=1e-12,
      atol=1e-12)

      vf = u"497070.0m/s"
      m = u"0.02eV"
      # freq      = u"25.0THz"
      # emax      = u"0.5MV/cm"
      t2 = Inf * u"1.0s"
      t1 = Inf * u"1.0s"
      e = uconvert(u"C", 1u"eV" / 1u"V")

      γ = M / ζ
      ω = 2m / (M * Unitful.ħ)
      freq = uconvert(u"THz", ω / 2π)
      σ = uconvert(u"fs", 1 / freq)
      emax = uconvert(u"MV/cm", ω * m / (vf * e * γ))
      us = scaledriving_frequency(freq, vf)
      df = GaussianAPulse(us, σ, freq, emax)
      h = GappedDirac(us, m, vf)
      l = TwoBandDephasingLiouvillian(h, Inf, Inf)

      dt = 0.005
      # ts      = -5df.σ:dt:5df.σ
      ts = -5df.σ:dt:-df.σ
      kxmax = 10 * Damysos.maximum_kdisplacement(df, ts)[1]
      kymax = 1.0
      dkx = 2kxmax / 1_200
      dky = 1.0

      pars = NumericalParams2d(dkx, dky, kxmax, kymax, dt, -5df.σ, rtol, atol)
      obs = [Velocity(h), Occupation(h)]

      id = "M=$(round(M,sigdigits=3))_zeta=$(round(ζ,sigdigits=3))"
      name = "Simulation(2d)_$id"
      dpath = joinpath(datapath_base, subpath, name)
      ppath = joinpath(plotpath_base, subpath, name)

      return Simulation(l, df, pars, obs, us, 2, id, dpath, ppath)
end

const Ms    = (collect(0.1:0.1:1.0)..., collect(2.0:1.0:5)...)
const zetas = (collect(0.1:0.1:1.0)..., collect(2.0:1.0:5)...)
const I     = CartesianIndices((length(Ms),length(zetas)))

function parse_cmdargs()
      s = ArgParseSettings()
      @add_arg_table! s begin
            "parameterset"
                  help = "Index of the parameter set, int from 1 to $(length(I))"
                  arg_type = Int
                  required = true
            "--verbose", "-v"
                  help = "Verbosity: show debug messages in stderr if enabled"
                  nargs = 0
            "--atolgoal", "-a"
                  help = "Goal for absolute tolerance"
                  arg_type = Float64
                  default = 1e-12
            "--rtolgoal", "-r"
                  help = "Goal for relative tolerance"
                  arg_type = Float64
                  default = 1e-4
      end
      return parse_args(s)
end

const cmdargs = parse_cmdargs()
const i = I[cmdargs["parameterset"]]
const M = Ms[i[1]]
const z = zetas[i[2]]
const atolgoal = cmdargs["atolgoal"]
const rtolgoal = cmdargs["rtolgoal"]

@info "ζ = $z γ = $M"

const sim = make_system(
      z,
      M,
      "hhgjl/inter-intra-cancellation/m_zeta_grid/dt";
      rtol=rtolgoal / 2,
      atol=atolgoal / 2)

const logpath  = dirname(sim.datapath)
const loglevel = cmdargs["verbose"] ? Logging.Debug : Logging.Info 

ensurepath(logpath)
global_logger(make_teelogger(logpath, "convergence-tests-z=$z-M=$M.log",loglevel))
@info "Logging to \"$logpath\"/convergence-tests-z=$z-M=$M.log"

const test = ConvergenceTest(
      sim,
      LinearChunked(128),
      PowerLawTest(:dt, 0.5),
      atolgoal,
      rtolgoal,
      60 * 10,
      32)
res = run!(test)
str = repr("text/plain", res)
@info """
Result for ζ = $z γ = $M

$str


"""

@info "Plotting for last sim"
plotdata(res.test.completedsims[end])

@info "Finished"