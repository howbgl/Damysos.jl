using Damysos,Unitful,LoggingExtras,Dates,Formatting,Distributed,DifferentialEquations

import Damysos.getshortname
import Damysos.ensurepath
import Damysos.getdipoles_x


function runtest(sim::Simulation{T},kys::AbstractVector{T}) where T
      p              = getparams(sim)

      γ1              = oneunit(T) / p.t1
      γ2              = oneunit(T) / p.t2

      nkx            = p.nkx
      kx_samples     = p.kxsamples
      tsamples       = p.tsamples
      tspan          = (tsamples[1],tsamples[end])

      a              = get_vecpotx(sim.drivingfield)
      f              = get_efieldx(sim.drivingfield)
      ϵ              = getϵ(sim.hamiltonian)

      dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

      rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(oneunit(T)-cc)
      rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


      @inline function rhs!(du,u,p,t)
            @inbounds for i in 1:nkx
            du[i] = rhs_cc(t,u[i],u[i+nkx],kx_samples[i],p[1])
            end

            @inbounds for i in nkx+1:2nkx
            du[i] = rhs_cv(t,u[i-nkx],u[i],kx_samples[i-nkx],p[1])
            end
      end

      u0    = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
      prob  = ODEProblem(rhs!,u0,tspan)
      sols  = pmap(
            ky -> solve(remake(prob,p=[ky]),reltol=p.rtol,abstol=p.atol),
            kys
      )

      return sols
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
const kxmax   = 1750.0
const dky     = 1.0
const kymax   = 100.0

const us      = scaledriving_frequency(freq,vf)
const h       = GappedDirac(us,m,vf,t1,t2)
const df      = GaussianPulse(us,σ,freq,emax)
const pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
const obs     = [Velocity(h)]

# const id      = sprintf1("%x",hash([h,df,pars,obs,us]))
const id      = "bigref"
const name    = "Simulation{$(typeof(h.Δ))}(2d)reference"
const dpath   = "test/reference/big"
const ppath   = dpath

const sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)
