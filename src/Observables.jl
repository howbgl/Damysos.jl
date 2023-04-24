
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

@inline function normalize!(v::Velocity{T},norm::T) where {T<:Real}
    v.vx ./= norm
    v.vy ./= norm
    v.vxinter ./= norm
    v.vyinter ./= norm
    v.vxintra ./= norm
    v.vyintra ./= norm
end

function calcobs_k1d!(sim::Simulation{T},v::Velocity{T},sol,ky::T,
                    vxinter_k::Array{T},vxintra_k::Array{T},
                    vyinter_k::Array{T},vyintra_k::Array{T}) where {T<:Real}

    p     = getparams(sim)
    kx    = p.kxsamples
    ax    = get_vecpotx(sim.drivingfield)
    ay    = get_vecpoty(sim.drivingfield)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)
    vy_cc = getvy_cc(sim.hamiltonian)
    vy_vc = getvy_vc(sim.hamiltonian)

    if sim.dimensions==1        
        for i in 1:length(sol.t)
            vxinter_k[:,i] = real.((2 .* sol[1:p.nkx,i] .- 1) .* 
                                vx_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            vxintra_k[:,i] = 2 .* real.(vx_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])
        end
    elseif sim.dimensions==2
        for i in 1:length(sol.t)
            vxinter_k[:,i] = real.((2 .* sol[1:p.nkx,i] .- 1) .* 
                                vx_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            vxintra_k[:,i] = 2 .* real.(vx_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])
            vyinter_k[:,i] = real.((2 .* sol[1:p.nkx,i] .- 1) .* 
                                vy_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            vyintra_k[:,i] = 2 .* real.(vy_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])       
        end
    end
end

function integrate1d_obs(sim::Simulation{T},v::Velocity{T},sol,ky::T,
                    moving_bz::Array{T}) where {T<:Real}

    p           = getparams(sim)
    vxintra_k   = zeros(T,p.nkx,length(sol.t))
    vxinter_k   = zeros(T,p.nkx,length(sol.t))
    vxintra     = zeros(T,length(sol.t))
    vxinter     = zeros(T,length(sol.t))
    vx          = zeros(T,length(sol.t))
    vyintra_k   = zeros(T,p.nkx,length(sol.t))
    vyinter_k   = zeros(T,p.nkx,length(sol.t))
    vyintra     = zeros(T,length(sol.t))
    vyinter     = zeros(T,length(sol.t))
    vy          = zeros(T,length(sol.t))
    
    calcobs_k1d!(sim,v,sol,ky,vxinter_k,vxintra_k,vyinter_k,vyintra_k)

    vxintra = trapz((p.kxsamples,:),vxintra_k .* moving_bz)
    vxinter = trapz((p.kxsamples,:),vxinter_k .* moving_bz)    
    vyintra = trapz((p.kxsamples,:),vyintra_k .* moving_bz)
    vyinter = trapz((p.kxsamples,:),vyinter_k .* moving_bz)
    @. vx   = vxinter + vxintra
    @. vy   = vyinter + vyintra

    return Velocity(vx,vxintra,vxinter,vy,vyintra,vyinter)
end

function integrate2d_obs!(vels::Vector{Velocity{T}},
    kysamples::Vector{T}) where {T<:Real}

    vx      = trapz((:,hcat(kysamples)),hcat([v.vx for v in vels]...))
    vxintra = trapz((:,hcat(kysamples)),hcat([v.vxintra for v in vels]...))
    vxinter = trapz((:,hcat(kysamples)),hcat([v.vxinter for v in vels]...))
    vy      = trapz((:,hcat(kysamples)),hcat([v.vy for v in vels]...))
    vyintra = trapz((:,hcat(kysamples)),hcat([v.vyintra for v in vels]...))
    vyinter = trapz((:,hcat(kysamples)),hcat([v.vyinter for v in vels]...))

    return Velocity(vx,vxintra,vxinter,vy,vyintra,vyinter)
end


struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
end
function Occupation(h::Hamiltonian{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0))
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

function integrate2d_obs!(occs::Vector{Occupation{T}},
    kysamples::Vector{T}) where {T<:Real}

    cbocc  = trapz((:,hcat(kysamples)),hcat([o.cbocc for o in occs]...))
    return Occupation(cbocc)
end



function calc_obs_k1d(sim::Simulation{T},sol,ky::T) where {T<:Real}

    p              = getparams(sim)
    ax             = get_vecpotx(sim.drivingfield)
    ay             = get_vecpoty(sim.drivingfield)
    moving_bz      = zeros(T,p.nkx,length(sol.t))

    sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
    bzmask1d(kx)   = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))

    if sim.dimensions==1
        
        for i in 1:length(sol.t)
            moving_bz[:,i] .= bzmask1d.(p.kxsamples .- ax(sol.t[i]))
        end
    elseif sim.dimensions==2
        kxs = p.kxsamples
        bzmask2d(kx,ky)= bzmask1d(kx)*sig((ky-p.bz[3])/(2*p.dky)) * sig((p.bz[4]-ky)/(2*p.dky))
        for i in 1:length(sol.t)
            moving_bz[:,i] .= bzmask2d.(kxs .- ax(sol.t[i]),ky - ay(sol.t[i]))
        end

    end

    obs     = [integrate1d_obs(sim,o,sol,ky,moving_bz) for o in sim.observables]
    return obs
end
