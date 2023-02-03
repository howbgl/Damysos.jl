module Damysos


using Unitful,Accessors,Trapz,DifferentialEquations,Interpolations,Plots,DSP,DataFrames,Random,CSV

export Hamiltonian,GappedDirac,getϵ,getdx_cc,getdx_cv,getdx_vc,getdx_vv,getdipoles_x,getvels_x
export getvx_cc,getvx_cv,getvx_vc,getvx_vv
export DrivingField,GaussianPulse,get_efield,get_vecpot
export NumericalParameters,NumericalParams2d,NumericalParams1d
export Simulation,Ensemble,getparams,parametersweep
export Observable,Velocity,Occupation,Timesteps,getnames_obs
export UnitScaling,semiclassical_interband_range
export run_simulation,run_simulation1d,run_simulation2d

abstract type SimulationComponent{T} end
abstract type Hamiltonian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end
abstract type Observable{T}             <: SimulationComponent{T} end

struct UnitScaling{T<:Real}
    timescale::Unitful.Time{T}
    lengthscale::Unitful.Length{T}
end

struct Simulation{T<:Real}
    hamiltonian::Hamiltonian{T}
    drivingfield::DrivingField{T}
    numericalparams::NumericalParameters{T}
    observables::Vector{Observable{T}}
    unitscaling::UnitScaling{T}
    dimensions::UInt8
    name::String
    datapath::String
    plotpath::String
    function Simulation{T}(h,df,p,obs,us,d,name,dpath,ppath) where {T<:Real}
        if p isa NumericalParams1d{T} && d!=1
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(1),name,dpath,ppath)
        elseif p isa NumericalParams2d{T} && d!=2
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(2),name,dpath,ppath)
        else
            new(h,df,p,obs,us,d,name,dpath,ppath)
        end
    end
end
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,
    name::String,dpath::String,ppath::String) where {T<:Real} = Simulation{T}(h,df,p,obs,us,UInt8(abs(d)),name,dpath,ppath)
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,name) where {T<:Real} = Simulation(h,df,p,obs,us,d,String(name),"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer) where {T<:Real} = Simulation(h,df,p,obs,us,d,randstring(4),"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")

function Base.show(io::IO,::MIME"text/plain",s::Simulation{T}) where {T}
    print(io,"Simulation{$T} ($(s.dimensions)d) with components{$T}:\n")
    for n in fieldnames(Simulation{T})
        if !(n == :dimensions)
            print(io,"  ")
            Base.show(io,MIME"text/plain"(),getfield(s,n))
            print(io,'\n')
        end
    end
end

function getshortname(sim::Simulation{T}) where {T<:Real}
    return "Simulation{$T}($(sim.dimensions)d)" * split("_$(sim.hamiltonian)",'{')[1] * split("_$(sim.drivingfield)",'{')[1]
end

function saveparams(sim::Simulation{T}) where {T<:Real}
    CSV.write(sim.datapath * getfilename(sim) * "_params.csv",DataFrame([getparams(sim)]))
end

getparams(sim::Simulation{T}) where {T<:Real} = merge((bz=(-sim.numericalparams.kxmax + 1.3*sim.drivingfield.eE/sim.drivingfield.ω, sim.numericalparams.kxmax - 1.3*sim.drivingfield.eE/sim.drivingfield.ω),),
    getparams(sim.hamiltonian),getparams(sim.drivingfield),getparams(sim.numericalparams))
getnames_obs(sim::Simulation{T}) where {T<:Real} = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation{T}) where {T<:Real} = vcat(arekresolved.(sim.observables)...)
getfilename(sim::Simulation{T}) where {T<:Real}  = getshortname(sim)*'_'*sim.name


struct Ensemble{T<:Real}
    simlist::Vector{Simulation{T}}
    name::String
    datapath::String
    plotpath::String
end
Ensemble(sl::Vector{Simulation{T}},name::String) where {T<:Real} = Ensemble(sl,name,"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")
Ensemble(sl::Vector{Simulation{T}},name) where {T<:Real}         = Ensemble(sl,String(name)) 
Ensemble(sl::Vector{Simulation{T}}) where {T<:Real}              = Ensemble(sl,"defaultens") 

Base.size(a::Ensemble)                  = (size(a.simlist))
Base.setindex!(a::Ensemble,v,i::Int)    = (a.simlist[i] = v)
Base.getindex(a::Ensemble,i::Int)       = a.simlist[i]
function Base.show(io::IO,::MIME"text/plain",e::Ensemble{T}) where {T}
    print(io,"Ensemble{$T} of Simulations{$T}:\n")
    for i in 1:length(e.simlist)
        print(io,"  #$i\n","  ")
        Base.show(io,MIME"text/plain"(),e.simlist[i])
        print(io,"\n")
    end
end

function getshortname(ens::Ensemble{T}) where {T<:Real}
    return "Ensemble{$T}[$(length(ens.simlist))]($(ens.simlist[1].dimensions)d)" * split("_$(ens.simlist[1].hamiltonian)",'{')[1] * split("_$(ens.simlist[1].drivingfield)",'{')[1]
end

getfilename(ens::Ensemble{T}) where {T<:Real} = getshortname(ens) * ens.name

getshortname(obs::Observable{T}) where {T<:Real} = split("$obs",'{')[1]

function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, range::AbstractVector{T}) where {T<:Real}
    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    if comp isa Hamiltonian{T}
        for i in 1:length(sweeplist)
            new_h           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(new_h,sim.drivingfield,sim.numericalparams,sim.observables,sim.unitscaling,sim.dimensions)
        end
    elseif comp isa DrivingField{T}
        for i in 1:length(sweeplist)
            new_df          = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,new_df,sim.numericalparams,sim.observables,sim.unitscaling,sim.dimensions)
        end
    elseif comp isa NumericalParameters{T}
        for i in 1:length(sweeplist)
            new_p           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,sim.drivingfield,new_p,sim.observables,sim.unitscaling,sim.dimensions)
        end
    else
        return nothing
    end
    return Ensemble(sweeplist)
end

function Base.show(io::IO,::MIME"text/plain",c::SimulationComponent{T}) where {T}
    pars = getparams(c)
    print(io,split("$c",'{')[1],":  $pars")
end

struct GappedDirac{T<:Real} <: Hamiltonian{T}
    Δ::T
    t2::T
    GappedDirac{T}(Δ,t2) where {T<:Real} = new(Δ,t2) 
end
GappedDirac(Δ::T,t2::T) where {T<:Real}      = GappedDirac{T}(Δ,t2)
GappedDirac(Δ::Real,t2::Real)                = GappedDirac(promote(Δ,t2)...)
function GappedDirac(mass::Unitful.Energy{T},fermivelocity::Unitful.Velocity{T},
                    dephasing::Unitful.Time{T}) where{T<:Real}
    tc = uconvert(u"fs",0.1*Unitful.ħ/mass)
    lc = uconvert(u"nm",fermivelocity*tc)
    us = UnitScaling(tc,lc)
    return GappedDirac(us,mass,fermivelocity,dephasing)
end
function GappedDirac(us::UnitScaling{T},mass::Unitful.Energy{T},
    fermivelocity::Unitful.Velocity{T},dephasing::Unitful.Time{T}) where{T<:Real}
    Δ   = uconvert(Unitful.NoUnits,mass*us.timescale/Unitful.ħ)
    t2  = uconvert(Unitful.NoUnits,dephasing/us.timescale)
    return us,GappedDirac(Δ,t2)
end

getparams(h::GappedDirac{T}) where {T<:Real} = (Δ=h.Δ,t2=h.t2)

getϵ(h::GappedDirac{T})     where {T<:Real}  = (kx,ky) -> sqrt(kx^2+ky^2+h.Δ^2)
getdx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> ky * (1.0 -h. h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)
getdx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx + 2.0im*ky)
getdx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx - 2.0im*ky)
getdx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -ky * (1.0 -h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)

getvx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> kx/sqrt(kx^2+ky^2+h.Δ^2)
getvx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*ky) / (kx + 1.0im*ky)
getvx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*ky) / (kx - 1.0im*ky)
getvx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -kx/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> ky/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*kx) / (kx + 1.0im*ky)
getvy_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*kx) / (kx - 1.0im*ky)
getvy_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -ky/sqrt(kx^2+ky^2+h.Δ^2)

getdipoles_x(h::GappedDirac{T}) where {T<:Real}  = (getdx_cc(h),getdx_cv(h),getdx_vc(h),getdx_vv(h))
getvels_x(h::GappedDirac{T}) where {T<:Real}     = (getvx_cc(h),getvx_cv(h),getvx_vc(h),getvx_vv(h))  


struct GaussianPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
    GaussianPulse{T}(σ,ω,eE) where {T<:Real} = new(σ,ω,eE)
end
GaussianPulse(σ::T,ω::T,eE::T) where {T<:Real} = GaussianPulse{T}(σ,ω,eE)
GaussianPulse(σ::Real,ω::Real,eE::Real)        = GaussianPulse(promote(σ,ω,eE)...)
function GaussianPulse(us::UnitScaling{T},standard_dev::Unitful.Time{T},
                    frequency::Unitful.Frequency{T},fieldstrength::Unitful.EField{T}) where{T<:Real}
    σ   = uconvert(Unitful.NoUnits,standard_dev/us.timescale)
    ω   = uconvert(Unitful.NoUnits,2π*frequency*us.timescale)
    e   = uconvert(u"C",1u"eV"/1u"V")
    eE  = uconvert(Unitful.NoUnits,e*us.timescale*us.lengthscale*fieldstrength/Unitful.ħ)
    return GaussianPulse(σ,ω,eE)
end

getparams(df::GaussianPulse{T}) where {T<:Real}  = (σ=df.σ,ν=df.ω/2π,ω=df.ω,eE=df.eE)

function get_efield(df::GaussianPulse{T}) where {T<:Real}
    return t-> df.eE * (t * cos(df.ω*t) + df.σ^2 * df.ω * sin(df.ω*t)) * exp(-t^2 / (2df.σ^2)) / (df.ω*df.σ^2) 
end
function get_vecpot(df::GaussianPulse{T}) where {T<:Real}
    return t -> df.eE * cos(df.ω*t) * exp(-t^2 / (2df.σ^2)) / df.ω
end


struct NumericalParams2d{T<:Real} <: NumericalParameters{T}
    dkx::T
    dky::T
    kxmax::T
    kymax::T
    dt::T
    t0::T
    NumericalParams2d{T}(dkx,dky,kxmax,kymax,dt,t0) where {T<:Real} = new(dkx,dky,kxmax,kymax,dt,t0) 
end
NumericalParams2d(dkx::T,dky::T,kxmax::T,kymax::T,dt::T,t0::T) where {T<:Real}    = NumericalParams2d{T}(dkx,dky,kxmax,kymax,dt,t0)
NumericalParams2d(dkx::Real,dky::Real,kxmax::Real,kymax::Real,dt::Real,t0::Real)  = NumericalParams2d(promote(dkx,dky,kxmax,kymax,dt,t0)...)

getparams(p::NumericalParams2d{T}) where {T<:Real} = (dkx=p.dkx,dky=p.dky,kxmax=p.kxmax,kymax=p.kymax,dt=p.dt,t0=p.t0,
nkx=2*Int(cld(p.kxmax,p.dkx)),nky=2*Int(cld(p.kymax,p.dky)),nt=2*Int(cld(abs(p.t0),p.dt)),
tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))),kysamples=LinRange(-p.kymax,p.kymax,2*Int(cld(p.kymax,p.dky))))

struct NumericalParams1d{T<:Real} <: NumericalParameters{T}
    dkx::T
    kxmax::T
    dt::T
    t0::T
    NumericalParams1d{T}(dkx,kxmax,dt,t0) where{T<:Real} = new(dkx,kxmax,dt,t0)
end
NumericalParams1d(dkx::T,kxmax::T,dt::T,t0::T) where {T<:Real} = NumericalParams1d{T}(dkx,kxmax,dt,t0)
NumericalParams1d(dkx::Real,kxmax::Real,dt::Real,t0::Real)     = NumericalParams1d(promote(dkx,kxmax,dt,t0)...)

getparams(p::NumericalParams1d{T}) where {T<:Real} = (dkx=p.dkx,kxmax=p.kxmax,
nkx=2*Int(cld(p.kxmax,p.dkx)),nt=2*Int(cld(abs(p.t0),p.dt)),dt=p.dt,
tsamples=LinRange(-abs(p.t0),abs(p.t0),2*Int(cld(abs(p.t0),p.dt))),
kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))))


struct Velocity{T<:Real} <: Observable{T}
    dummy::T
end

getnames_obs(v::Velocity{T}) where {T<:Real} = ["vx","vxintra","vxinter"]
getparams(v::Velocity{T}) where {T<:Real}    = getnames_obs(v)
arekresolved(v::Velocity{T}) where {T<:Real} = [false, false, false]


function calcobs_k1d!(sim::Simulation{T},v::Velocity{T},sol,vxinter_k::Array{T},vxintra_k::Array{T}) where {T<:Real}
    p     = getparams(sim)
    a     = get_vecpot(sim.drivingfield)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)

    for i in 1:length(sol.t)
        vxinter_k[:,i] = real.((2.0 .* sol[1:p.nkx,i] .- 1.0) .* vx_cc.(p.kxsamples .- a(sol.t[i]),0.0))
        vxintra_k[:,i] = 2.0 .* real.(vx_vc.(p.kxsamples .-a(sol.t[i]),0.0) .* sol[(p.nkx+1):end,i])
    end
end

function integrate_obs(sim::Simulation{T},v::Velocity{T},sol,moving_bz::Array{T}) where {T<:Real}
    p           = getparams(sim)

    vxintra_k   = zeros(T,p.nkx,length(sol.t))
    vxinter_k   = zeros(T,p.nkx,length(sol.t))
    vxintra     = zeros(T,length(sol.t))
    vxinter     = zeros(T,length(sol.t))
    vx          = zeros(T,length(sol.t))
    
    calcobs_k1d!(sim,v,sol,vxinter_k,vxintra_k)

    vxintra = trapz((p.kxsamples,:),vxintra_k .* moving_bz)
    vxinter = trapz((p.kxsamples,:),vxinter_k .* moving_bz)
    @. vx   = vxinter + vxintra

    return vx,vxintra,vxinter
end


struct Occupation{T<:Real} <: Observable{T}
    dummy::T
end
getnames_obs(occ::Occupation{T}) where {T<:Real} = ["cb_occ", "cb_occ_k"]
getparams(occ::Occupation{T}) where {T<:Real}    = getnames_obs(occ)
arekresolved(occ::Occupation{T}) where {T<:Real} = [false, true]

function calcobs_k1d!(sim::Simulation{T},occ::Occupation{T},sol,occ_k::Array{T},occ_k_itp::Array{T}) where {T<:Real}
    p        = getparams(sim)
    a        = get_vecpot(sim.drivingfield)
    
    occ_k   .= real.(sol[1:p.nkx,:])

    for i in 1:length(sol.t)
        kxt_range = LinRange(p.kxsamples[1]-a(sol.t[i]),p.kxsamples[end]-a(sol.t[i]), p.nkx)
        itp       = interpolate((kxt_range,),real(sol[1:p.nkx,i]), Gridded(Linear()))
        for j in 2:size(occ_k_itp)[1]-1
            occ_k_itp[j,i] = itp(p.bz[1] + j*p.dkx)
        end
   end
end

function integrate_obs(sim::Simulation{T},o::Occupation{T},sol,moving_bz::Array{T}) where {T<:Real}

    p           = getparams(sim)
    nkx_bz      = Int(cld(2*p.bz[2],p.dkx))

    occ_k_itp   = zeros(T,nkx_bz,length(sol.t))
    occ_k       = zeros(T,p.nkx,length(sol.t))
    occ         = zeros(T,length(sol.t))
    
    calcobs_k1d!(sim,o,sol,occ_k,occ_k_itp)

    occ         = trapz((p.kxsamples,:),occ_k .* moving_bz)

    return occ,occ_k_itp
end


function calc_obs(sim::Simulation{T},sol) where {T<:Real}

    p              = getparams(sim)
    a              = get_vecpot(sim.drivingfield)
    moving_bz      = zeros(T,p.nkx,length(sol.t))
    
    sig(x)         = 0.5*(1.0+tanh(x/2.0)) # logistic function 1/(1+e^(-t)) = (1 + tanh(x/2))/2
    bzmask(kx)     = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))
    
    for i in 1:length(sol.t)
        moving_bz[:,i] .= bzmask.(p.kxsamples .- a(sol.t[i]))
    end

    obs     = []
    for i in 1:length(sim.observables)
        append!(obs,integrate_obs(sim,sim.observables[i],sol,moving_bz))
    end
    return obs
end

function run_simulation1d(sim::Simulation{T},ky::T;rtol=1e-10,atol=1e-10,savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p              = getparams(sim)
    
    γ              = 1.0 / p.t2
    nkx            = p.nkx
    kx_samples     = p.kxsamples
    tsamples       = p.tsamples
    tspan          = (tsamples[1],tsamples[end])
    
    a              = get_vecpot(sim.drivingfield)
    f              = get_efield(sim.drivingfield)
    ϵ              = getϵ(sim.hamiltonian)

    dcc,dcv,dvc,dvv          = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cv,kx,ky)     = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky))
    rhs_cv(t,cc,cv,kx,ky)  = (-γ - 2.0im * ϵ(kx-a(t),ky)) * cv - 1.0im * f(t) * (2.0 * dvv(kx-a(t),ky) * cv + dcv(kx-a(t),ky) * (2.0cc - 1.0))


    @inline function rhs!(du,u,p,t)
         for i in 1:nkx
              du[i] = rhs_cc(t,u[i+nkx],kx_samples[i],ky)
         end
    
         for i in nkx+1:2nkx
              du[i] = rhs_cv(t,u[i-nkx],u[i],kx_samples[i-nkx],ky)
         end
         return
    end

    u0             = zeros(T,2*nkx) .+ im .* zeros(T,2*nkx)
    prob           = ODEProblem(rhs!,u0,tspan)
    sol            = solve(prob;saveat=tsamples,reltol=rtol,abstol=atol,kwargs...)
    
    obs = calc_obs(sim,sol)

    if savedata == true
        Damysos.savedata(sim,obs)
    end

    if saveplots == true
        Damysos.plotdata(sim)
    end

    return obs
end

function run_simulation2d(sim::Simulation{T};savedata=true,saveplots=true,kwargs...) where {T<:Real}

    p         = getparams(sim)
    total_obs = []
    last_obs  = run_simulation1d(sim,p.kysamples[1];savedata=false,saveplots=false,kwargs...)

    for i in 2:p.nky
        if mod(i,10)==0
            println(100.0i/p.nky,"%")
        end
        obs = run_simulation1d(sim,p.kysamples[i];savedata=false,saveplots=false,kwargs...)
        for o in obs
            push!(total_obs,trapz((:,hcat(p.kysamples[i-1],p.kysamples[i])),hcat(last_obs[j],o)))
        end
        last_obs = deepcopy(obs)
    end

    if savedata == true
        Damysos.savedata(sim,total_obs)
    end

    if saveplots == true
        Damysos.plotdata(sim,datapath)
    end

    return total_obs
end

function run_simulation(sim::Simulation{T};kwargs...) where {T<:Real}
    if sim.dimensions==1
        obs = run_simulation1d(sim,0.0;kwargs...)
    elseif sim.dimensions==2
        obs = run_simulation2d(sim;kwargs...)
    end
    return obs
end

function run_simulation(ens::Ensemble{T};savedata=true,saveplots=true,
                makecombined_plots=true,kwargs...) where {T<:Real}
    
    datapath = "/home/how09898/phd/data/hhgjl/" * lowercase(getshortname(ens))
    plotpath = "/home/how09898/phd/plots/hhgjl/" * lowercase(getshortname(ens))

    allobs = []

    for i in eachindex(ens.simlist)
        obs = run_simulation(ens.simlist[i];savedata=savedata,saveplots=saveplots,
                            datapath=datapath,plotpath=plotpath,kwargs...)
        push!(allobs,obs)
    end

    if makecombined_plots == true
        Damysos.plotdata(ens,allobs)
    end

    return allobs
end

function savedata(sim::Simulation{T},obs) where {T<:Real}

    if !isdir(sim.datapath)
        mkpath(sim.datapath)
    end
    dat         = DataFrame()
    names       = getnames_obs(sim)
    arekres     = arekresolved(sim)
    filename    = getfilename(sim)

    if length(eachindex(names)) == length(eachindex(obs))   
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(names[i]),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end 
    else
        println("length(eachindex(names)) != length(eachindex(obs)) in savedata(sim::Simulation{T},obs)")
        println("Using numbers as names instead...")
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(i),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end
    end

    CSV.write(sim.datapath*filename*".csv",dat)
    println("Saved Simulation data at ",sim.datapath*filename,".csv")

    return nothing
end

function plotdata(ens::Ensemble{T},obs;plotpath="/home/how09898/phd/plots/hhgjl/",kwargs...) where {T<:Real}
    ensemblename = lowercase(getshortname(ens))
    if !isdir(plotpath*ensemblename)
        mkpath(plotpath*ensemblename)
    end

    if length(eachindex(ens.simlist)) == length(eachindex(obs))

        allobsnames = getnames_obs(ens[1])
        arekres     = arekresolved(ens[1])
        for i in eachindex(allobsnames)
            if arekres[i] == true
                continue
            else
                figs    = plot();
                fftfigs = plot();
                for j in eachindex(obs)
                    p           = getparams(ens[j])
                    plot!(figs,p.tsamples,obs[j][i],label=String(j))

                    pdg         = periodogram(obs[j][i],nfft=8*length(obs[j][i]),fs=1/p.dt,window=blackman)
                    maxharm     = maximum(pdg.freq)/p.ν
                    plot!(fftfigs,
                        1/p.ν .* pdg.freq, 
                        pdg.power,
                        yscale=:log10,
                        xticks=0:5:maxharm,
                        xminorticks=0:maxharm,
                        xminorgrid=true,
                        xgridalpha=0.3,
                        label=String(j))
                end
                savefig(figs,plotpath*ensemblename*'/'*allobsnames[i]*".pdf")
                savefig(fftfigs,plotpath*ensemblename*'/'*allobsnames[i]*"_spec.pdf")
            end
        end
        
    else
        println("length(eachindex(ens.simlist)) != length(eachindex(obs)) in savedata(ens::Ensemble{T},obs,datapath=...)")
        println("Aborting...")
    end
end

function plotdata(sim::Simulation{T};fftwindow=blackman,kwargs...) where {T<:Real}
    p           = getparams(sim)
    filename    = getfilename(sim)
    alldata     = DataFrame(CSV.File(sim.datapath*filename*".csv"))

    println("Skip plotting k-resolved data for now...")

    for obs in sim.observables
        obspath = sim.plotpath*filename*'/'*getshortname(obs)
        if !isdir(obspath)
            mkpath(obspath)
        end
        plotdata(obs,alldata,p,obspath;fftwindow=fftwindow,kwargs...)
        
    end

    println("Saved plots at ",sim.plotpath*filename*"/")
    return nothing
end

function plotdata(obs::Observable{T},alldata::DataFrame,p,plotpath::String;fftwindow=blackman,kwargs...) where {T<:Real}

    nonkresolved_obs = []
    periodograms     = []

    for i in eachindex(getnames_obs(obs))
        arekres     = arekresolved(obs)
        obsnames    = getnames_obs(obs)
        if arekres[i] == true
            continue
        else
            push!(nonkresolved_obs,obsnames[i])
            data = getproperty(alldata,Symbol(obsnames[i]))
            push!(periodograms,periodogram(data,nfft=8*length(data),fs=1/p.dt,window=fftwindow))
        end
    end
    nonkresolved_data = select(alldata,nonkresolved_obs)
    maxharm = maximum(periodograms[1].freq)/p.ν
    fftfig  = plot(1/p.ν .* periodograms[1].freq, 
                hcat([x.power for x in periodograms]...),
                yscale=:log10,
                xticks=0:5:maxharm,
                xminorticks=0:maxharm,
                xminorgrid=true,
                xgridalpha=0.3,
                label=permutedims(nonkresolved_obs))
    fig     = plot(p.tsamples,Matrix(nonkresolved_data),label=permutedims(nonkresolved_obs))

    savefig(fig,plotpath*".pdf")
    savefig(fftfig,plotpath*"_spec.pdf")

    return nothing
end

function semiclassical_interband_range(h::GappedDirac,df::GaussianPulse)
    ϵ        = getϵ(h)
    ωmin     = 2.0*ϵ(0.0,0.0)
    kmax     = df.eE/df.ω
    ωmax     = 2.0*ϵ(kmax,0.0)
    min_harm = ωmin/df.ω
    max_harm = ωmax/df.ω
    println("Approximate range of semiclassical interband: ",min_harm," to ",
            max_harm," (harmonic number)")
end

end
