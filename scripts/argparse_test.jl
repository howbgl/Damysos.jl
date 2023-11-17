using ArgParse,Distributed

function parse_cmdargs()
      argsettings = ArgParseSettings()
      @add_arg_table argsettings begin
            "--heap-size-hint", "-s"
                  help = "RAM hint (per worker!) passed to all child processes & threads"
                  arg_type = String
                  default = "2G"
            "--worker-processes", "-p"
                  help = "number of processes to be launched with addprocs"
                  arg_type = Int
                  default  = 0
            "--kxbatch-basesize", "-x"
                  help = "number of kx-vertices solved & integrated at once"
                  arg_type = Int
                  default  = 64
            "--kybatch-basesize", "-y"
                  help = "number of 1d simulation running in a (possibly parallel) batch"
                  arg_type = Int
                  default  = 64
            "--no-plots"  
                  help = "do not generate plots"
                  action = :store_true
            "--no-data"
                  help = "do not save numerical data to disk, i.e. only plots or dry-run"
                  action = :store_true    
      end

      return parse_args(argsettings)
end

parsed_args = parse_cmdargs()
sizehint    = parsed_args["heap-size-hint"]
nworkers    = parsed_args["worker-processes"]
addprocs(nworkers;exeflags=["--project", "--heap-size-hint=$sizehint"])

using Damysos,Unitful,LoggingExtras,Dates,Formatting,TerminalLoggers

import Damysos.getshortname

const vf        = u"4.3e5m/s"
const freq      = u"2THz"
const m         = u"50.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) 
const t2        = tcycle / 4            
const t1        = Inf*u"1s"
const σ         = u"2000.0fs"

# converged at
# dt = 0.001
# dkx = 0.1
# dky = 1.0
# kxmax = 
# kymax = 

const dt      = 0.001
const dkx     = 0.1
const kxmax   = 1.0
const dky     = 1.0
const kymax   = 50.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const name    = "Simulation{$(typeof(h.Δ))}(2d)"*getshortname(h)*"_"*getshortname(df)*"_$id"
const dpath   = "/home/how09898/phd/data/hhgjl/argparse_test/"*name
const ppath   = "/home/how09898/phd/plots/hhgjl/argparse_test/"*name

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
const ens     = parametersweep(sim,sim.numericalparams,:dkx,[0.5,0.25,0.1])

global_logger(TerminalLogger())

@info "# $PROGRAM_FILE"

run_simulation!(ens;
      kxbatch_basesize=parsed_args["kxbatch-basesize"],
      maxparallel_ky=parsed_args["kybatch-basesize"],
      savedata=!parsed_args["no-data"],
      saveplots=!parsed_args["no-plots"])
