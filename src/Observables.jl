import Base.empty,Base.zero

export Observable,Velocity,Occupation,getnames_obs,zero!,resize

sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 

struct Velocity{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
    vy::Vector{T}
    vyintra::Vector{T}
    vyinter::Vector{T}
end
function Velocity(h::Hamiltonian{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0),
                    Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0))
end

function Velocity(p::NumericalParameters{T}) where {T<:Real}

    params      = getparams(p)
    nt          = params.nt

    vxintra     = zeros(T,nt)
    vxinter     = zeros(T,nt)
    vx          = zeros(T,nt)
    vyintra     = zeros(T,nt)
    vyinter     = zeros(T,nt)
    vy          = zeros(T,nt)

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter)
end

function resize(v::Velocity{T},p::NumericalParameters{T})  where {T<:Real}
    return Velocity(p)
end

function empty(v::Velocity) 
    return Velocity(v)
end


getnames_obs(v::Velocity{T}) where {T<:Real} = ["vx","vxintra","vxinter","vy","vyintra",
                                                "vyinter"]
getparams(v::Velocity{T}) where {T<:Real}    = getnames_obs(v)
arekresolved(v::Velocity{T}) where {T<:Real} = [false,false,false,false,false,false]


@inline function addto!(v::Velocity{T},vtotal::Velocity{T}) where {T<:Real}
    vtotal.vx .= vtotal.vx .+ v.vx
    vtotal.vy .= vtotal.vy .+ v.vy
    vtotal.vxinter .= vtotal.vxinter .+ v.vxinter
    vtotal.vyinter .= vtotal.vyinter .+ v.vyinter
    vtotal.vxintra .= vtotal.vxintra .+ v.vxintra
    vtotal.vyintra .= vtotal.vyintra .+ v.vyintra
end

@inline function copyto!(vdest::Velocity,vsrc::Velocity)
    vdest.vx        .= vsrc.vx
    vdest.vxintra   .= vsrc.vxintra
    vdest.vxinter   .= vsrc.vxinter
    vdest.vy        .= vsrc.vy
    vdest.vyintra   .= vsrc.vyintra
    vdest.vyintra   .= vsrc.vyintra
end

@inline function normalize!(v::Velocity{T},norm::T) where {T<:Real}
    v.vx ./= norm
    v.vy ./= norm
    v.vxinter ./= norm
    v.vyinter ./= norm
    v.vxintra ./= norm
    v.vyintra ./= norm
end

function zero(v::Velocity{T}) where {T<:Real}
    
    vxintra     = zero(v.vxintra)
    vxinter     = zero(v.vxinter)
    vx          = zero(v.vx)
    vyintra     = zero(v.vyintra)
    vyinter     = zero(v.vyinter)
    vy          = zero(v.vy)
    return Velocity(vx,vxintra,vxinter,vy,vyintra,vyinter)
end

function getfuncs(sim::Simulation,v::Velocity)
    df = sim.drivingfield
    h  = sim.hamiltonian
    return [get_vecpotx(df),get_vecpoty(df),getvx_cc(h),getvx_vc(h),getvx_vv(h),
            getvy_cc(h),getvy_vc(h),getvy_vv(h)]
end


@inline function vintra(kx::T,ky::T,ρcc::Complex{T},vcc,vvv) where {T<:Real}
    return vintra(kx,ky,real(ρcc),vcc,vvv)
end
@inline function vintra(kx::T,ky::T,ρcc::T,vcc,vvv) where {T<:Real}
    return vcc(kx,ky)*ρcc + vvv(kx,ky)*(oneunit(T)-ρcc)
end

@inline function vinter(kx::T,ky::T,ρcv::Complex{T},vvc) where {T<:Real}
    return 2 * real(vvc(kx,ky) * ρcv)
end

function integrateobs_kxbatch_add!(
    sim::Simulation{T},
    v::Velocity{T},
    sol,
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    funcs) where {T<:Real}

    ax,ay,vx_cc,vx_vc,vx_vv,vy_cc,vy_vc,vy_vv = funcs

    ts    = getparams(sim).tsamples
    nkx   = length(kxsamples)
    kxt   = zeros(T,nkx)
    vbuff = zeros(T,nkx)

    for (i,t) in enumerate(ts)
        kxt             .= kxsamples .- ax(t)
        ρcc             = @view sol[1:nkx,i]
        ρcv             = @view sol[nkx+1:2nkx,i]

        vbuff           .= vintra.(kxt,ky,ρcc,vx_cc,vx_vv)
        v.vxintra[i]    += trapz(kxsamples,moving_bz[:,i] .* vbuff)

        vbuff           .= vinter.(kxt,ky,ρcv,vx_vc)
        v.vxinter[i]    += trapz(kxsamples,moving_bz[:,i] .* vbuff)

        vbuff           .= vintra.(kxt,ky,ρcc,vy_cc,vy_vv)
        v.vyintra[i]    += trapz(kxsamples,moving_bz[:,i] .* vbuff)

        vbuff           .= vinter.(kxt,ky,ρcv,vy_vc)
        v.vyinter[i]    += trapz(kxsamples,moving_bz[:,i] .* vbuff)
    end
    return v
end

function integrateobs_kxbatch!(
    sim::Simulation{T},
    v::Velocity{T},
    sol,
    ky::T,
    moving_bz::Array{T}) where {T<:Real}

    p     = getparams(sim)
    kxt   = zeros(T,p.nkx)
    ax    = get_vecpotx(sim.drivingfield)
    ay    = get_vecpoty(sim.drivingfield)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vv = getvx_vv(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)
    vy_cc = getvy_cc(sim.hamiltonian)
    vy_vc = getvy_vc(sim.hamiltonian)
    vy_vv = getvy_vv(sim.hamiltonian)

    for (i,t) in enumerate(p.tsamples)
        kxt             .= p.kxsamples .- ax(t)
        v.vxintra[i]    = trapz(p.kxsamples,moving_bz[:,i] .* vintra.(kxt,ky,sol[1:p.nkx,i],vx_cc,vx_vv))
        v.vxinter[i]    = trapz(p.kxsamples,moving_bz[:,i] .* vinter.(kxt,ky,sol[p.nkx+1:2p.nkx,i],vx_vc))
        v.vyintra[i]    = trapz(p.kxsamples,moving_bz[:,i] .* vintra.(kxt,ky,sol[1:p.nkx,i],vy_cc,vy_vv))
        v.vyinter[i]    = trapz(p.kxsamples,moving_bz[:,i] .* vinter.(kxt,ky,sol[p.nkx+1:2p.nkx,i],vy_vc))
    end

    @. v.vx   = v.vxinter + v.vxintra
    @. v.vy   = v.vyinter + v.vyintra

    return v
end

function integrateobs(
    vels::Vector{Velocity{T}},
    vertices::Vector{T}) where {T<:Real}

    vdest = zero(vels[1])
    return integrateobs!(vels,vdest,vertices)
end

function integrateobs!(
    vels::Vector{Velocity{T}},
    vdest::Velocity{T},
    vertices::Vector{T}) where {T<:Real}

    vdest.vxintra .= trapz((:,hcat(vertices)),hcat([v.vxintra for v in vels]...))
    vdest.vxinter .= trapz((:,hcat(vertices)),hcat([v.vxinter for v in vels]...))
    vdest.vyintra .= trapz((:,hcat(vertices)),hcat([v.vyintra for v in vels]...))
    vdest.vyinter .= trapz((:,hcat(vertices)),hcat([v.vyinter for v in vels]...))

    @. vdest.vx   = vdest.vxintra + vdest.vxinter
    @. vdest.vy   = vdest.vyintra + vdest.vyinter
end

function integrateobs_threaded!(
    vels::Vector{Velocity{T}},
    vdest::Velocity{T},
    vertices::Vector{T}) where {T<:Real}

    @floop ThreadedEx() for i in 1:length(vdest.vx)
        vdest.vxintra[i] = trapz(vertices,[v.vxintra[i] for v in vels])
        vdest.vxinter[i] = trapz(vertices,[v.vxinter[i] for v in vels])
        vdest.vyintra[i] = trapz(vertices,[v.vyintra[i] for v in vels])
        vdest.vyinter[i] = trapz(vertices,[v.vyinter[i] for v in vels])
    end

    @. vdest.vx   = vdest.vxintra + vdest.vxinter
    @. vdest.vy   = vdest.vyintra + vdest.vyinter    
end

struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
end
function Occupation(h::Hamiltonian{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0))
end
function Occupation(p::NumericalParameters{T}) where {T<:Real}
    return Occupation(zeros(T,getparams(p).nt))
end

function resize(o::Occupation{T},p::NumericalParameters{T}) where {T<:Real}
    return Occupation(p)
end

function empty(o::Occupation)
    return Occupation(o)
end

getnames_obs(occ::Occupation{T}) where {T<:Real} = ["cbocc", "cbocck"]
getparams(occ::Occupation{T}) where {T<:Real}    = getnames_obs(occ)
arekresolved(occ::Occupation{T}) where {T<:Real} = [false, true]

@inline function addto!(o::Occupation{T},ototal::Occupation{T}) where {T<:Real}
    ototal.cbocc .= ototal.cbocc .+ o.cbocc
end

@inline function copyto!(odest::Occupation,osrc::Occupation)
    odest.cbocc .= osrc.cbocc
end

@inline function normalize!(o::Occupation{T},norm::T) where {T<:Real}
    o.cbocc ./= norm
end

function zero(o::Occupation{T}) where {T<:Real}
    cbocc = zero(o.cbocc)
    return Occupation(cbocc)
end

function integrateobs_kxbatch!(sim::Simulation{T},o::Occupation{T},sol,ky::T,
                    moving_bz::Array{T}) where {T<:Real}

    p           = getparams(sim)
    nkx_bz      = Int(cld(2*p.bz[2],p.dkx))

    occ_k_itp   = zeros(T,nkx_bz,length(sol.t))
    occ_k       = zeros(T,p.nkx,length(sol.t))
    occ         = zeros(T,length(sol.t))
    
    calcobs_k1d!(sim,o,sol,occ_k,occ_k_itp)

    occ         = trapz((p.kxsamples,:),occ_k .* moving_bz)

    return Occupation(occ)
end

function integrateobs(
    occs::Vector{Occupation{T}},
    vertices::Vector{T}) where {T<:Real}

    cbocc  = trapz((:,hcat(vertices)),hcat([o.cbocc for o in occs]...))
    return Occupation(cbocc)
end


function getmovingbz(sim::Simulation{T}) where {T<:Real}
    p              = getparams(sim)
    return getmovingbz(sim,p.kxsamples)
end

function getmovingbz(sim::Simulation{T},kxsamples::AbstractVector{T}) where {T<:Real}
    p              = getparams(sim)
    nkx            = length(kxsamples)
    ax             = get_vecpotx(sim.drivingfield)
    ay             = get_vecpoty(sim.drivingfield)
    moving_bz      = zeros(T,nkx,p.nt)
    bzmask1d(kx)   = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))

    for i in 1:p.nt
        moving_bz[:,i] .= bzmask1d.(kxsamples .- ax(p.tsamples[i]))
    end
    return moving_bz    
end

function integrateobs!(
    observables::Vector{Vector{Observable{T}}},
    observables_dest::Vector{Observable{T}},
    vertices::AbstractVector{T}) where {T<:Real}

    for (i,odest) in enumerate(observables_dest) 
        integrateobs!([o[i] for o in observables],odest,collect(vertices))
    end    
end

function integrateobs_threaded!(
    observables::Vector{Vector{Observable{T}}},
    observables_dest::Vector{Observable{T}},
    vertices::AbstractVector{T}) where {T<:Real}

    for (i,odest) in enumerate(observables_dest) 
        integrateobs_threaded!([o[i] for o in observables],odest,collect(vertices))
    end    
end

function integrateobs_kxbatch_add!(
    sim::Simulation{T},
    sol,
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    obsfuncs) where {T<:Real}

    for (o,funcs) in zip(sim.observables,obsfuncs)
        integrateobs_kxbatch_add!(sim,o,sol,kxsamples,ky,moving_bz,funcs)
    end        
end