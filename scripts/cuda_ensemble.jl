using Damysos,Unitful,LoggingExtras,Dates,Formatting,DifferentialEquations,DiffEqGPU

import Damysos.getshortname
import Damysos.ensurepath


function buildensemble_linear(sim::Simulation;kchunksize=4096)

      p                   = getparams(sim)
      kxs                 = p.kxsamples
      kys                 = p.kysamples
      γ1                  = one(p.t1) / p.t1
      γ2                  = one(p.t1) / p.t2    
      a                   = get_vecpotx(sim.drivingfield)
      f                   = get_efieldx(sim.drivingfield)
      ϵ                   = getϵ(sim.hamiltonian)
      dcc,dcv,dvc,dvv     = getdipoles_x(sim.hamiltonian)
  
      rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(one(t)-cc)
      rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - im * f(t) * 
          ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2cc - one(t)))
      
      @inline function rhs(u,p,t)
          return SA[rhs_cc(t,u[1],u[2],p[1],p[2]),rhs_cv(t,u[1],u[2],p[1],p[2])]
      end
  
      u0             = SA[zero(im*p.t1),zero(im*p.t1)]
      tspan          = (p.tsamples[1],p.tsamples[end])
      prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))
  
      ensprob = EnsembleProblem(
          prob,
          prob_func   = (prob,i,repeat) -> remake(prob,p = getkgrid_point(i,kxs,kys)),
          safetycopy  = false)
  
      return ensprob
  end
  

const vf        = u"4.3e5m/s"
const freq      = u"5THz"
const m         = u"20.0meV"
const emax      = u"0.1MV/cm"
const tcycle    = uconvert(u"fs",1/freq) # 100 fs
const t2        = tcycle / 4             # 25 fs
const t1        = Inf*u"1s"
const σ         = u"800.0fs"

# converged at
# dt = 0.01
# dkx = 1.0
# dky = 1.0
# kxmax = 175
# kymax = 100

const dt      = 0.01
const dkx     = 1.0
const kxmax   = 175.0
const dky     = 1.0
const kymax   = 100.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const id      = "cudatest1"
const name    = "Simulation{$(typeof(h.Δ))}(2d)reference"
const dpath   = "test/cuda"
const ppath   = dpath

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)

ensurepath(sim.plotpath)
const info_filelogger  = FileLogger(joinpath(sim.plotpath,sim.id*"_$(now()).log"))
const info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
const all_filelogger   = FileLogger(joinpath(sim.plotpath,sim.id*"_$(now())_debug.log"))
const console_logger   = ConsoleLogger(stdout)
const tee_logger       = TeeLogger(info_logger,all_filelogger,console_logger)

@info "Logging to $(joinpath(sim.plotpath,getshortname(sim)*"s(now()).log")) " *
      "and $(joinpath(sim.plotpath,getshortname(sim)*"_$(now())_debug.log"))"

global_logger(tee_logger)
@info "$(now())\nOn $(gethostname()):"

const ens   = buildensemble_linear(sim)
const p     = getparams(sim)
sols        = solve(ens,EnsembleGPUArray();trajectories=p.nkx*p.nky)

@info "$(now()): calculation finished."
