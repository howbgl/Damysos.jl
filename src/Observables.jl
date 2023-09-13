struct Velocity{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
    vy::Vector{T}
    vyintra::Vector{T}
    vyinter::Vector{T}
    vxintra_k::Matrix{T}
    vxinter_k::Matrix{T}
    vyintra_k::Matrix{T}
    vyinter_k::Matrix{T}
end
function Velocity(h::Hamiltonian{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Vector{T}(undef,0),
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0))
end

function Velocity(p::NumericalParameters{T}) where {T<:Real}

    params      = getparams(p)
    nkx         = params.nkx
    nt          = params.nt

    vxintra_k   = zeros(T,nkx,nt)
    vxinter_k   = zeros(T,nkx,nt)
    vxintra     = zeros(T,nt)
    vxinter     = zeros(T,nt)
    vx          = zeros(T,nt)
    vyintra_k   = zeros(T,nkx,nt)
    vyinter_k   = zeros(T,nkx,nt)
    vyintra     = zeros(T,nt)
    vyinter     = zeros(T,nt)
    vy          = zeros(T,nt)

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter,
                    vxintra_k,
                    vxinter_k,
                    vyintra_k,
                    vyinter_k)
end

function resize(v::Velocity{T},p::NumericalParameters{T})  where {T<:Real}
    return Velocity(p)
end

getnames_obs(v::Velocity{T}) where {T<:Real} = ["vx","vxintra","vxinter","vy","vyintra",
                                                "vyinter"]
getparams(v::Velocity{T}) where {T<:Real}    = getnames_obs(v)
arekresolved(v::Velocity{T}) where {T<:Real} = [false,false,false,false,false,false]

@inline function addto!(vtotal::Velocity{T},v::Velocity{T}) where {T<:Real}

    v.vx        .= v.vx .+ vtotal.vx
    v.vy        .= v.vy .+ vtotal.vy
    v.vxinter   .= v.vxinter .+ vtotal.vxinter
    v.vyinter   .= v.vyinter .+ vtotal.vyinter
    v.vxintra   .= v.vxintra .+ vtotal.vxintra
    v.vyintra   .= v.vyintra .+ vtotal.vyintra
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
    vxintra_k   = zero(v.vxintra_k)
    vxinter_k   = zero(v.vxinter_k)
    vyintra_k   = zero(v.vyintra_k)
    vyinter_k   = zero(v.vyinter_k)

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter,
                    vxintra_k,
                    vxinter_k,
                    vyintra_k,
                    vyinter_k)
end

function zero!(v::Velocity{T}) where {T<:Real}
    
    v.vx        .= zero(T)
    v.vxintra   .= zero(T)
    v.vxinter   .= zero(T)
    v.vy        .= zero(T)
    v.vyintra   .= zero(T)
    v.vyinter   .= zero(T)
    v.vxintra_k .= zero(T)
    v.vxinter_k .= zero(T)
    v.vyintra_k .= zero(T)
    v.vyinter_k .= zero(T)
end

function calcobs_k1d!(sim::Simulation{T},v::Velocity{T},sol,ky::T) where {T<:Real}
    
    p     = getparams(sim)
    kx    = p.kxsamples
    ts    = p.tsamples
    ax    = get_vecpotx(sim.drivingfield)
    ay    = get_vecpoty(sim.drivingfield)
    kxt   = zeros(T,p.nkx)
    kyt   = zeros(T,p.nkx)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vv = getvx_vv(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)
    vy_cc = getvy_cc(sim.hamiltonian)
    vy_vc = getvy_vc(sim.hamiltonian)
    vy_vv = getvy_vv(sim.hamiltonian)

    for i in eachindex(ts)
        kxt                 .= kx .- ax(ts[i])
        kyt                 .= ky .- ay(ts[i])
        v.vxintra_k[:,i]    .= real.(sol[1:p.nkx,i] .* vx_cc.(kxt,kyt) .+
                            (1 .- sol[1:p.nkx,i]) .*vx_vv.(kxt,kyt))
        v.vxinter_k[:,i]    .= 2 .* real.(vx_vc.(kxt,kyt) .* sol[(p.nkx+1):end,i])
    end

    if sim.dimensions==2
        for i in eachindex(ts)
            kxt               .= kx .- ax(ts[i])
            kyt               .= ky .- ay(ts[i])
            v.vyintra_k[:,i]  .= real.(sol[1:p.nkx,i] .* vy_cc.(kxt,kyt) .+
                                (1 .- sol[1:p.nkx,i]) .*vy_vv.(kxt,kyt))
            v.vyinter_k[:,i]  .= 2 .* real.(vy_vc.(kxt,kyt) .* sol[(p.nkx+1):end,i])   
        end
    end
end

function integrate1d_obs(sim::Simulation{T},v::Velocity{T},sol,ky::T,
                    moving_bz::Array{T}) where {T<:Real}

    kxs = getparams(sim).kxsamples
    
    zero!(v)
    
    calcobs_k1d!(sim,v,sol,ky)

    v.vxintra .= trapz((kxs,:),v.vxintra_k .* moving_bz)
    v.vxinter .= trapz((kxs,:),v.vxinter_k .* moving_bz)    
    v.vyintra .= trapz((kxs,:),v.vyintra_k .* moving_bz)
    v.vyinter .= trapz((kxs,:),v.vyinter_k .* moving_bz)
    @. v.vx   = v.vxinter + v.vxintra
    @. v.vy   = v.vyinter + v.vyintra

    return v
end

function integrate2d_obs(vels::Vector{Velocity{T}},
    kysamples::Vector{T}) where {T<:Real}

    v           = zero(vels[1])
    v.vx        .= trapz((:,hcat(kysamples)),hcat([v.vx for v in vels]...))
    v.vxintra   .= trapz((:,hcat(kysamples)),hcat([v.vxintra for v in vels]...))
    v.vxinter   .= trapz((:,hcat(kysamples)),hcat([v.vxinter for v in vels]...))
    v.vy        .= trapz((:,hcat(kysamples)),hcat([v.vy for v in vels]...))
    v.vyintra   .= trapz((:,hcat(kysamples)),hcat([v.vyintra for v in vels]...))
    v.vyinter   .= trapz((:,hcat(kysamples)),hcat([v.vyinter for v in vels]...))

    return v
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

getnames_obs(occ::Occupation{T}) where {T<:Real} = ["cbocc", "cbocck"]
getparams(occ::Occupation{T}) where {T<:Real}    = getnames_obs(occ)
arekresolved(occ::Occupation{T}) where {T<:Real} = [false, true]

@inline function addto!(o::Occupation{T},ototal::Occupation{T}) where {T<:Real}
    ototal.cbocc .= ototal.cbocc .+ o.cbocc
end

@inline function normalize!(o::Occupation{T},norm::T) where {T<:Real}
    o.cbocc ./= norm
end

function zero(o::Occupation{T}) where {T<:Real}
    cbocc = zero(o.cbocc)
    return Occupation(cbocc)
end

function calcobs_k1d!(sim::Simulation{T},occ::Occupation{T},sol,
                    occ_k::Array{T},occ_k_itp::Array{T}) where {T<:Real}
    p        = getparams(sim)
    a        = get_vecpotx(sim.drivingfield)
    
    occ_k   .= real.(sol[1:p.nkx,:])

    for i in 1:length(sol.t)
        kxt_range = LinRange(p.kxsamples[1]-a(sol.t[i]),p.kxsamples[end]-a(sol.t[i]), p.nkx)
        itp       = interpolate((kxt_range,),real(sol[1:p.nkx,i]), Gridded(Linear()))
        for j in 2:size(occ_k_itp)[1]-1
            occ_k_itp[j,i] = itp(p.bz[1] + j*p.dkx)
        end
   end
end

function integrate1d_obs(sim::Simulation{T},o::Occupation{T},sol,ky::T,
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

function integrate2d_obs(occs::Vector{Occupation{T}},
    kysamples::Vector{T}) where {T<:Real}

    cbocc  = trapz((:,hcat(kysamples)),hcat([o.cbocc for o in occs]...))
    return Occupation(cbocc)
end



function calc_obs_k1d(sim::Simulation{T},sol,ky::T) where {T<:Real}

    p              = getparams(sim)
    ax             = get_vecpotx(sim.drivingfield)
    ay             = get_vecpoty(sim.drivingfield)
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
    
    obs     = [integrate1d_obs(sim,o,sol,ky,moving_bz) for o in sim.observables]
    return obs
end
