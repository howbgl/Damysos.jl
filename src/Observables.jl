import Base.empty,Base.zero

export Observable,Velocity,Occupation,getnames_obs,zero!,resize

struct Velocity{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
    vy::Vector{T}
    vyintra::Vector{T}
    vyinter::Vector{T}
end

function Velocity(::Velocity{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0))
end

# backwards compatibility
function Velocity(h::Hamiltonian{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0))
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

@inline function addto!(vdest::Velocity{T},vsrc::Velocity{T}) where {T<:Real}

    vdest.vx        .= vsrc.vx .+ vdest.vx
    vdest.vy        .= vsrc.vy .+ vdest.vy
    vdest.vxinter   .= vsrc.vxinter .+ vdest.vxinter
    vdest.vyinter   .= vsrc.vyinter .+ vdest.vyinter
    vdest.vxintra   .= vsrc.vxintra .+ vdest.vxintra
    vdest.vyintra   .= vsrc.vyintra .+ vdest.vyintra
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

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter)
end

function zero!(v::Velocity{T}) where {T<:Real}
    
    v.vx        .= zero(T)
    v.vxintra   .= zero(T)
    v.vxinter   .= zero(T)
    v.vy        .= zero(T)
    v.vyintra   .= zero(T)
    v.vyinter   .= zero(T)
end


function calc_obs_mode!(
    v::Velocity{T},
    sol::Matrix{Complex{T}},
    tsamples::AbstractVector{T},
    kx::T,
    ky::T,
    obs_funcs) where {T<:Real}

    vx_cc,vx_cv,vx_vc,vx_vv,vy_cc,vy_cv,vy_vc,vy_vv,ax,ay,fx,fy = obs_funcs

    kxt       = kx .- ax.(tsamples)
    v.vxintra .= real.(sol[1,:] .* vx_cc.(kxt,ky) .+ (1 .- sol[1,:]) .* vx_vv.(kxt,ky))
    v.vxinter .= 2.0 .* real.(vx_vc.(kxt,ky) .* sol[2,:])
    v.vyintra .= real.(sol[1,:] .* vy_cc.(kxt,ky) .+ (1 .- sol[1,:]) .* vy_vv.(kxt,ky))
    v.vyinter .= 2.0 .* real.(vy_vc.(kxt,ky) .* sol[2,:])
end



function integrate_obs!(
    vels::Vector{Velocity{T}},
    vdest::Velocity{T},
    vertices::Vector{T}) where {T<:Real}

    vdest.vx        .= trapz((:,hcat(vertices)),hcat([v.vx for v in vels]...))
    vdest.vxintra   .= trapz((:,hcat(vertices)),hcat([v.vxintra for v in vels]...))
    vdest.vxinter   .= trapz((:,hcat(vertices)),hcat([v.vxinter for v in vels]...))
    vdest.vy        .= trapz((:,hcat(vertices)),hcat([v.vy for v in vels]...))
    vdest.vyintra   .= trapz((:,hcat(vertices)),hcat([v.vyintra for v in vels]...))
    vdest.vyinter   .= trapz((:,hcat(vertices)),hcat([v.vyinter for v in vels]...))
end

function integrate_obs_add!(
    vels::Vector{Velocity{T}},
    vdest::Velocity{T},
    kysamples::Vector{T}) where {T<:Real}
    
    vdest.vx        .+= trapz((:,hcat(kysamples)),hcat([v.vx for v in vels]...))
    vdest.vxintra   .+= trapz((:,hcat(kysamples)),hcat([v.vxintra for v in vels]...))
    vdest.vxinter   .+= trapz((:,hcat(kysamples)),hcat([v.vxinter for v in vels]...))
    vdest.vy        .+= trapz((:,hcat(kysamples)),hcat([v.vy for v in vels]...))
    vdest.vyintra   .+= trapz((:,hcat(kysamples)),hcat([v.vyintra for v in vels]...))
    vdest.vyinter   .+= trapz((:,hcat(kysamples)),hcat([v.vyinter for v in vels]...))
end

function get_funcs(v::Velocity{T},sim::Simulation{T}) where {T<:Real}
    
    funcs = let h = sim.hamiltonian, df=sim.drivingfield
        (getvels_x(h)...,getvels_y(h)...,getfields(df)...)
    end
    return funcs
end



struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
end
function Occupation(::Occupation{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0))
end
# backwards compatibility
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

@inline function addto!(odest::Occupation{T},osrc::Occupation{T}) where {T<:Real}
    odest.cbocc .= odest.cbocc .+ osrc.cbocc
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

function calcobs_k1d!(occ::Occupation{T},sol,occ_k::Array{T},occ_k_itp::Array{T}) where {T<:Real}
#     p        = getparams(sim)
#     a        = get_vecpotx(sim.drivingfield)
    
#     occ_k   .= real.(sol[1:p.nkx,:])

#     for i in 1:length(sol.t)
#         kxt_range = LinRange(p.kxsamples[1]-a(sol.t[i]),p.kxsamples[end]-a(sol.t[i]), p.nkx)
#         itp       = interpolate((kxt_range,),real(sol[1:p.nkx,i]), Gridded(Linear()))
#         for j in 2:size(occ_k_itp)[1]-1
#             occ_k_itp[j,i] = itp(p.bz[1] + j*p.dkx)
#         end
#    end
end

function integrate1d_obs(
    o::Occupation{T},
    sol,
    p::NamedTuple,
    ky::T,
    moving_bz::Array{T},
    obs_funcs::Tuple) where {T<:Real}

    oresult = zero(o)
    integrate1d_obs!(oresult,sol,p,ky,moving_bz,obs_funcs)
    return oresult
end


function integrate1d_obs!(
    o::Occupation{T},
    sol,
    p::NamedTuple,
    ky::T,
    moving_bz::Array{T},
    obs_funcs::Tuple) where {T<:Real}

    @warn "Occupation not implemented yet"

    return o
end

function integrate_obs!(
    occs::Vector{Occupation{T}},
    odest::Occupation{T},
    vertices::Vector{T}) where {T<:Real}

    odest.cbocc .= trapz((:,hcat(vertices)),hcat([o.cbocc for o in occs]...))
end

function integrate_obs_add!(
    occs::Vector{Occupation{T}},
    odest::Occupation{T},
    kysamples::Vector{T}) where {T<:Real}

    odest.cbocc .+= trapz((:,hcat(kysamples)),hcat([o.cbocc for o in occs]...))
end


function get_funcs(o::Occupation{T},sim::Simulation{T}) where {T<:Real}
    
    return (nothing,)
end

function get_movingbz(
    df::DrivingField{T},
    p::NamedTuple) where {T<:Real}
    
    ax             = get_vecpotx(df)
    # ay             = get_vecpoty(df)
    moving_bz      = zeros(T,p.nkx,p.nt)

    sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
    bzmask1d(kx)   = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))

    for i in eachindex(p.tsamples)
        moving_bz[:,i] .= bzmask1d.(p.kxsamples .- ax(p.tsamples[i]))
    end

    # if sim.dimensions==1
        
    #     for i in 1:length(sol.t)
    #         moving_bz[:,i] .= bzmask1d.(p.kxsamples .- ax(sol.t[i]))
    #     end
    # elseif sim.dimensions==2
    #     kxs = p.kxsamples
    #     bzmask2d(kx,ky)= bzmask1d(kx)*sig((ky-p.bz[3])/(2*p.dky)) * sig((p.bz[4]-ky)/(2*p.dky))
    #     for i in 1:length(sol.t)
    #         moving_bz[:,i] .= bzmask2d.(kxs .- ax(sol.t[i]),ky - ay(sol.t[i]))
    #     end

    # end

    return moving_bz
end

function calcobs_kybatch!(
    solutions::Vector{Matrix{Complex{T}}},
    obs_dest::Vector{Observable{T}},
    kxsamples::AbstractVector{T},
    kysamples::AbstractVector{T},
    tsamples::AbstractVector{T},
    moving_bz::Matrix{T},
    obs_funcs::Vector) where {T<:Real}

    for (o,funcs) in zip(obs_dest,obs_funcs)

        obs_batch = pmap(
            (u,ky) -> integrate1d_obs(o,u,kxsamples,tsamples,ky,moving_bz,funcs),
            solutions,
            kysamples)

        integrate2d_obs_add!(obs_batch,o,kysamples)
    end
end


function add_observables_batch!(
    obs_dest::Vector{Observable{T}},
    obs_batch::Vector{Vector{Observable{T}}}) where {T<:Real}

    for obs_src in obs_batch
        for (odest,osrc) in zip(obs_dest,obs_src)
            addto!(odest,osrc)
        end
    end    
end


function calc_allobs_1d!(
    observables::Vector{Observable{T}},
    sol::Matrix{Complex{T}},
    kxsamples::AbstractVector{T},
    ky::T,
    tsamples::AbstractVector{T},
    moving_bz::Matrix{T},
    obs_funcs::Vector) where {T<:Real}

    res = Vector{Observable{T}}(undef,0)
    
    for (o,funcs) in zip(observables,obs_funcs)
        push!(res,integrate1d_obs(o,sol,kxsamples,tsamples,ky,moving_bz,funcs))
    end

    return res
end
