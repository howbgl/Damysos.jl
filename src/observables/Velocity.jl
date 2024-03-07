export Velocity
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

function +(v1::Velocity,v2::Velocity)
    Velocity(
        v1.vx .+ v2.vx,
        v1.vxintra .+ v2.vxintra,
        v1.vxinter .+ v2.vxinter,
        v1.vy .+ v2.vy,
        v1.vyintra .+ v2.vyintra,
        v1.vyinter .+ v2.vyinter,)
end
function -(v1::Velocity,v2::Velocity)
    Velocity(
        v1.vx .- v2.vx,
        v1.vxintra .- v2.vxintra,
        v1.vxinter .- v2.vxinter,
        v1.vy .- v2.vy,
        v1.vyintra .- v2.vyintra,
        v1.vyinter .- v2.vyinter)
end
*(v::Velocity,x::Number) = Velocity(
    x .* v.vx,
    x .* v.vxintra,
    x .* v.vxinter,
    x .* v.vy,
    x .* v.vyintra,
    x .* v.vyinter)
*(x::Number,v::Velocity) = v * x

function zero(v::Velocity) 
    Velocity(
        zero(v.vx),
        zero(v.vxintra),
        zero(v.vxinter),
        zero(v.vy),
        zero(v.vyintra),
        zero(v.vyinter))
end

function buildobservable_expression(sim::Simulation,v::Velocity)
    
    h    = sim.liouvillian.hamiltonian
    df   = sim.drivingfield

    vxvc = vx_vc(h)
    vxcc = vx_cc(h)
    vxvv = vx_vv(h)
    vyvc = vy_vc(h)
    vycc = vy_cc(h)
    vyvv = vy_vv(h)

    ax  = vecpotx(df)
    ay  = vecpoty(df)
    
    vxintra_expr = :(real(u[1]) * ($vxcc - $vxvv))
    vxinter_expr = :(2real(u[2] * $vxvc))
    vyintra_expr = :(real(u[1]) * ($vycc - $vyvv))
    vyinter_expr = :(2real(u[2] * $vyvc))
    vel_expr     = :(SA[$vxintra_expr,$vxinter_expr,$vyintra_expr,$vyinter_expr])

    replace_expression!(vel_expr,:kx,:(kx-$ax))
    return vel_expr
end

function observable_from_data(sim::Simulation,v::Velocity,data)

    vxintra = empty(getkxsamples(sim.numericalparams))
    vxinter = empty(getkxsamples(sim.numericalparams))
    vyintra = empty(getkxsamples(sim.numericalparams))
    vyinter = empty(getkxsamples(sim.numericalparams))

    for v in data
        push!(vxintra,v[1])
        push!(vxinter,v[2])
        push!(vyintra,v[3])
        push!(vyinter,v[4])
    end
    return Velocity(vxinter .+ vxintra,vxintra,vxinter,vyinter .+ vyintra,vyintra,vyinter)
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