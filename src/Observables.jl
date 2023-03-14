
struct Velocity{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
end
function Velocity(h::Hamiltonian{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0))
end

getnames_obs(v::Velocity{T}) where {T<:Real} = ["vx","vxintra","vxinter"]
getparams(v::Velocity{T}) where {T<:Real}    = getnames_obs(v)
arekresolved(v::Velocity{T}) where {T<:Real} = [false, false, false]


function calcobs_k1d!(sim::Simulation{T},v::Velocity{T},sol,
                    vxinter_k::Array{T},vxintra_k::Array{T}) where {T<:Real}
    p     = getparams(sim)
    kx    = p.kxsamples
    a     = get_vecpotx(sim.drivingfield)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)

    for i in 1:length(sol.t)
        vxinter_k[:,i] = real.((2 .* sol[1:p.nkx,i] .- 1) .* vx_cc.(kx .- a(sol.t[i]),0))
        vxintra_k[:,i] = 2 .* real.(vx_vc.(kx .-a(sol.t[i]),0) .* sol[(p.nkx+1):end,i])
    end
end

function integrate1d_obs(sim::Simulation{T},v::Velocity{T},sol,
                    moving_bz::Array{T}) where {T<:Real}

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

    return Velocity(vx,vxintra,vxinter)
end

function integrate2d_obs!(sim::Simulation{T},vels::Vector{Velocity{T}},
    kysamples::Vector{T},total_obs::Vector{Observable{T}}) where {T<:Real}

    vx      = trapz((:,hcat(kysamples)),hcat([v.vx for v in vels]...))
    vxintra = trapz((:,hcat(kysamples)),hcat([v.vxintra for v in vels]...))
    vxinter = trapz((:,hcat(kysamples)),hcat([v.vxinter for v in vels]...))

    total_vel           = filter(x -> x isa Velocity,total_obs)[1]
    total_vel.vx        .+= vx
    total_vel.vxintra   .+= vxintra
    total_vel.vxinter   .+= vxinter
    return nothing
end


struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
    cbocck::Vector{Matrix{T}}
end
function Occupation(h::Hamiltonian{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0),Vector{Matrix{Float64}}(undef,0))
end

getnames_obs(occ::Occupation{T}) where {T<:Real} = ["cbocc", "cbocck"]
getparams(occ::Occupation{T}) where {T<:Real}    = getnames_obs(occ)
arekresolved(occ::Occupation{T}) where {T<:Real} = [false, true]

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

function integrate1d_obs(sim::Simulation{T},o::Occupation{T},sol,
                    moving_bz::Array{T}) where {T<:Real}

    p           = getparams(sim)
    nkx_bz      = Int(cld(2*p.bz[2],p.dkx))

    occ_k_itp   = zeros(T,nkx_bz,length(sol.t))
    occ_k       = zeros(T,p.nkx,length(sol.t))
    occ         = zeros(T,length(sol.t))
    
    calcobs_k1d!(sim,o,sol,occ_k,occ_k_itp)

    occ         = trapz((p.kxsamples,:),occ_k .* moving_bz)

    o.cbocc     = occ
    o.cbocck    = hcat(occ_k_itp) # o.cbocck must be matrix!

    return o
end


function calc_obs(sim::Simulation{T},sol) where {T<:Real}

    p              = getparams(sim)
    a              = get_vecpotx(sim.drivingfield)
    moving_bz      = zeros(T,p.nkx,length(sol.t))
    
    sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
    bzmask(kx)     = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))
    
    for i in 1:length(sol.t)
        moving_bz[:,i] .= bzmask.(p.kxsamples .- a(sol.t[i]))
    end

    obs     = [integrate1d_obs(sim,o,sol,moving_bz) for o in sim.observables]
    return obs
end
