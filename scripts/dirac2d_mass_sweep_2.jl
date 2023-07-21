using Damysos,Unitful,LoggingExtras,Dates,Formatting

import Damysos.getshortname
import Damysos.ensurepath

const vf        = u"4.3e5m/s"
const freq      = u"8THz"
const m         = u"100.0meV"
const emax      = u"0.3MV/cm"
const tcycle    = uconvert(u"fs",1/freq)
const t2        = 0.6tcycle
const σ         = u"100.0fs"

ensid   = "dirac2d_mass_sweep_2_100fs_8THz_0.3MVcm"

dt      = 0.1
dkx     = 0.002
kxmax   = 11.0
dky     = 0.01
kymax   = 2.0

simlist = Vector{Simulation{Float64}}(undef,0)

for x in LinRange(0.1,1.0,16)

    us,h    = scalegapped_dirac(x*m,vf,t2)
    df      = GaussianPulse(us,σ,freq,emax)
    pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
    obs     = [Velocity(h)]

    id      = sprintf1("%x",hash([h,df,pars,obs,us]))
    name    = "Simulation{$(typeof(h.Δ))}(2d)" * getshortname(h)*"_"*getshortname(df) * "_$id"
    dpath   = joinpath("/home/how09898/phd/data/hhgjl",ensid,name)
    ppath   = joinpath("/home/how09898/phd/plots/hhgjl",ensid,name)
    sim     = Simulation(h,df,pars,obs,us,2,id,dpath,ppath)

    push!(simlist,sim)
end

ens     = Ensemble(simlist,ensid)

ensurepath(ens.plotpath)
logger  = FileLogger(joinpath(ens.plotpath,getshortname(ens)*"_$(now()).log"))

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

results,time,rest... = @timed run_simulation!(ens;kxparallel=true)

@info "$(time/60.)min spent in run_simulation!(ens::Ensemble;...)"
@debug rest
@info "$(now()): calculation finished."
