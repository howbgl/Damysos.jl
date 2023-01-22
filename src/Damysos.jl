module Damysos


using Unitful,Accessors,Revise
export Hamiltonian,GappedDirac,getϵ,getdx_cc,getdx_cv,getdx_vc,getdx_vv,getdipoles_x,getvels_x
export getvx_cc,getvx_cv,getvx_vc,getvx_vv
export DrivingField,GaussianPulse,get_efield,get_vecpot
export NumericalParameters,NumericalParams2d,NumericalParams1d
export Simulation,Ensemble,getparams,parametersweep

abstract type SimulationComponent{T} end
abstract type Hamiltonian{T}            <: SimulationComponent{T} end
abstract type DrivingField{T}           <: SimulationComponent{T} end
abstract type NumericalParameters{T}    <: SimulationComponent{T} end

struct Simulation{T<:Real}
    hamiltonian::Hamiltonian{T}
    drivingfield::DrivingField{T}
    numericalparams::NumericalParameters{T}
    dimensions::UInt8
end
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},d::UInt8) where {T<:Real} = Simulation{T}(h,df,p,d)
Simulation(h::Hamiltonian{Real},df::DrivingField{Real},p::NumericalParameters{Real},d::UInt8)  = Simulation(promote(h,df,p)...,d)
Simulation(h::Hamiltonian,df::DrivingField,p::NumericalParameters,d::Integer) = Simulation(h,df,p,UInt8(d))
function Base.show(io::IO,::MIME"text/plain",s::Simulation{T}) where {T}
    print(io,"Simulation{$T} with components{$T}:\n")
    for n in fieldnames(Simulation{T})
        print("  ")
        Base.show(io,MIME"text/plain"(),getfield(s,n))
        print('\n')
    end
end

getparams(sim::Simulation{T}) where {T<:Real} = merge(getparams(sim.hamiltonian),getparams(sim.drivingfield),getparams(sim.numericalparams))

struct Ensemble{T<:Real}
    simlist::Vector{Simulation{T}}
end

Base.size(a::Ensemble)                  = (size(a.simlist))
Base.setindex!(a::Ensemble,v,i::Int)    = (a.simlist[i] = v)
Base.getindex(a::Ensemble,i::Int)       = a.simlist[i]
function Base.show(io::IO,::MIME"text/plain",e::Ensemble{T}) where {T}
    print(io,"Ensemble{$T} of Simulations{$T}:\n")
    for i in 1:length(e.simlist)
        print("  #$i\n","  ")
        Base.show(io,MIME"text/plain"(),e.simlist[i])
        print('\n')
    end
end

function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, range::AbstractVector{T}) where {T<:Real}
    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    if comp isa Hamiltonian{T}
        for i in 1:length(sweeplist)
            new_h           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(new_h,sim.drivingfield,sim.numericalparams)
        end
    elseif comp isa DrivingField{T}
        for i in 1:length(sweeplist)
            new_df          = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,new_df,sim.numericalparams)
        end
    elseif comp isa NumericalParameters{T}
        for i in 1:length(sweeplist)
            new_p           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,sim.drivingfield,new_p)
        end
    else
        return nothing
    end
    return Ensemble(sweeplist)
end

function Base.show(io::IO,::MIME"text/plain",c::SimulationComponent{T}) where {T}
    pars = getparams(c)
    print(io,split("$c",'(')[1],":  $pars")
end

struct GappedDirac{T<:Real} <: Hamiltonian{T}
    Δ::T
    t2::T
end
GappedDirac(Δ::T,t2::T) where {T<:Real} = GappedDirac{T}(Δ,t2)
GappedDirac(Δ::Real,t2::Real)           = GappedDirac(promote(Δ,t2)...)
getparams(h::GappedDirac{T}) where {T<:Real} = (Δ=h.Δ,t2=h.t2)

getϵ(h::GappedDirac{T})     where {T<:Real}  = (kx,ky) -> sqrt(kx^2+ky^2+h.Δ^2)
getdx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> ky * (1.0 -h. h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)
getdx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx + 2.0im*ky)
getdx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx - 2.0im*ky)
getdx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -ky * (1.0 -h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)
getvx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) ->  kx/sqrt(kx^2+ky^2+h.Δ^2)
getvx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) ->  (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*ky) / (kx + 1.0im*ky)
getvx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) ->  (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*ky) / (kx - 1.0im*ky)
getvx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) ->  -kx/sqrt(kx^2+ky^2+h.Δ^2)

getdipoles_x(h::GappedDirac{T}) where {T<:Real}  = (getdx_cc(h),getdx_cv(h),getdx_vc(h),getdx_vv(h))
getvels_x(h::GappedDirac{T}) where {T<:Real}     = (getvx_cc(h),getvx_cv(h),getvx_vc(h),getvx_vv(h))  


struct GaussianPulse{T<:Real} <: DrivingField{T}
    σ::T
    ω::T
    eE::T
end
GaussianPulse(σ::T,ω::T,eE::T) where {T<:Real} = GaussianPulse{T}(σ,ω,eE)
GaussianPulse(σ::Real,ω::Real,eE::Real)        = GaussianPulse(promote(σ,ω,eE)...)

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
end
NumericalParams2d(dkx::T,dky::T,kxmax::T,kymax::T,dt::T,t0::T) where {T<:Real}    = NumericalParams2d{T}(dkx,dky,kxmax,kymax,dt,t0)
NumericalParams2d(dkx::Real,dky::Real,kxmax::Real,kymax::Real,dt::Real,t0::Real)  = NumericalParams2d(promote(dkx,dky,kxmax,kymax,dt,t0)...)

getparams(p::NumericalParams2d{T}) where {T<:Real} = (dkx=p.dkx,dky=p.dky,kxmax=p.kxmax,kymax=p.kymax,dt=p.dt,t0=p.t0,kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))),kysamples=LinRange(-p.kymax,p.kymax,2*Int(cld(p.kymax,p.dky))))

struct NumericalParams1d{T<:Real} <: NumericalParameters{T}
    dkx::T
    kxmax::T
    dt::T
    t0::T
end
NumericalParams1d(dkx::T,kxmax::T,dt::T,t0::T) where {T<:Real} = NumericalParams1d{T}(dkx,kxmax,dt,t0)
NumericalParams1d(dkx::Real,kxmax::Real,dt::Real,t0::Real)     = NumericalParams1d(promote(dkx,kxmax,dt,t0)...)

getparams(p::NumericalParams1d{T}) where {T<:Real} = (dkx=p.dkx,kxmax=p.kxmax,dt=p.dt,t0=p.t0,kxsamples=kxsamples=LinRange(-p.kxmax,p.kxmax,2*Int(cld(p.kxmax,p.dkx))))


end
