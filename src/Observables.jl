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
    buffer::Array{T,3}
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
                    Matrix{T}(undef,0,0),
                    Array{T}(undef,0,0,0))
end
# for backwards compatibility:
function Velocity{T}(vx,vxintra,vxinter,vy,vyintra,vyinter) where {T<:Real}
    return Velocity(promote(vx,vxintra,vxinter,vy,vyintra,vyinter)...)
end
# for backwards compatibility:
function Velocity(vx::Vector{T},vxintra::Vector{T},vxinter::Vector{T},
                    vy::Vector{T},vyintra::Vector{T},vyinter::Vector{T}) where {T<:Real}
    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter,
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0),
                    Matrix{T}(undef,0,0),
                    Array{T}(undef,0,0,0))
end

function Velocity(p::NumericalParameters{T}) where {T<:Real}

    params      = getparams(p)
    nkx         = params.nkx
    nky         = p isa NumericalParams2d ? params.nky : 1
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
    buffer      = zeros(T,nky,nt,6)

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter,
                    vxintra_k,
                    vxinter_k,
                    vyintra_k,
                    vyinter_k,
                    buffer)
end

function resize(v::Velocity{T},p::NumericalParameters{T})  where {T<:Real}
    return Velocity(p)
end

getnames_obs(v::Velocity{T}) where {T<:Real} = ["vx","vxintra","vxinter","vy","vyintra",
                                                "vyinter"]
                                                
getparams(v::Velocity{T}) where {T<:Real}    = getnames_obs(v)
getshortname(v::Velocity{T}) where {T<:Real} = "Velocity"
arekresolved(v::Velocity{T}) where {T<:Real} = [false,false,false,false,false,false]

@inline function addto!(vtotal::Velocity{T},v::Velocity{T}) where {T<:Real}

    vtotal.vx        .= v.vx .+ vtotal.vx
    vtotal.vy        .= v.vy .+ vtotal.vy
    vtotal.vxinter   .= v.vxinter .+ vtotal.vxinter
    vtotal.vyinter   .= v.vyinter .+ vtotal.vyinter
    vtotal.vxintra   .= v.vxintra .+ vtotal.vxintra
    vtotal.vyintra   .= v.vyintra .+ vtotal.vyintra
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
    buffer      = zero(v.buffer)

    return Velocity(vx,
                    vxintra,
                    vxinter,
                    vy,
                    vyintra,
                    vyinter,
                    vxintra_k,
                    vxinter_k,
                    vyintra_k,
                    vyinter_k,
                    buffer)
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
    v.buffer    .= zero(T)
end

function calcobs_k1d!(sim::Simulation{T},v::Velocity{T},sol,ky::T) where {T<:Real}
    
    p     = getparams(sim)
    kx    = p.kxsamples
    ax    = get_vecpotx(sim.drivingfield)
    ay    = get_vecpoty(sim.drivingfield)
    vx_cc = getvx_cc(sim.hamiltonian)
    vx_vv = getvx_vv(sim.hamiltonian)
    vx_vc = getvx_vc(sim.hamiltonian)
    vy_cc = getvy_cc(sim.hamiltonian)
    vy_vc = getvy_vc(sim.hamiltonian)
    vy_vv = getvy_vv(sim.hamiltonian)

    if sim.dimensions==1        
        for i in 1:length(sol.t)
            v.vxintra_k[:,i] = real.(
                                sol[1:p.nkx,i] .* 
                                vx_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .+
                                (1 .- sol[1:p.nkx,i]) .*
                                vx_vv.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            v.vxinter_k[:,i] = 2 .* real.(vx_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])
        end
    elseif sim.dimensions==2
        for i in 1:length(sol.t)
            v.vxintra_k[:,i] = real.(
                                sol[1:p.nkx,i] .* 
                                vx_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .+
                                (1 .- sol[1:p.nkx,i]) .*
                                vx_vv.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            v.vxinter_k[:,i] = 2 .* real.(vx_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])
            v.vyintra_k[:,i] = real.(
                                sol[1:p.nkx,i] .* 
                                vy_cc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .+
                                (1 .- sol[1:p.nkx,i]) .*
                                vy_vv.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])))
            v.vyinter_k[:,i] = 2 .* real.(vy_vc.(kx .- ax(sol.t[i]),ky - ay(sol.t[i])) .* 
                                sol[(p.nkx+1):end,i])       
        end
    end
end

function integrate1d_obs!(sim::Simulation{T},v::Velocity{T},sol,ky::T,ky_index::Integer,
                    moving_bz::Array{T}) where {T<:Real}

    kxs = getparams(sim).kxsamples
    i   = ky_index
    zero!(v)
    
    calcobs_k1d!(sim,v,sol,ky)
    
    v.buffer[i,:,2] .= trapz((kxs,:),v.vxintra_k .* moving_bz)
    v.buffer[i,:,3] .= trapz((kxs,:),v.vxinter_k .* moving_bz)    
    v.buffer[i,:,5] .= trapz((kxs,:),v.vyintra_k .* moving_bz)
    v.buffer[i,:,6] .= trapz((kxs,:),v.vyinter_k .* moving_bz)
    v.buffer[i,:,1] .= v.buffer[i,:,2] .+ v.buffer[i,:,3]
    v.buffer[i,:,4] .= v.buffer[i,:,5] .+ v.buffer[i,:,6]
end

function finalize_obs1d!(s::Simulation{T},v::Velocity{T}) where {T<:Real}
    
    v.vx        .= v.buffer[1,:,1]
    v.vxintra   .= v.buffer[1,:,2]
    v.vxinter   .= v.buffer[1,:,3]
    v.vy        .= v.buffer[1,:,4]
    v.vyintra   .= v.buffer[1,:,5]
    v.vyinter   .= v.buffer[1,:,6]
end

function integrate2d_obs!(s::Simulation{T},v::Velocity{T}) where {T<:Real}
    
    kys         = collect(getparams(s).kysamples)
    v.vxintra   .= trapz(kys,v.buffer[:,:,2],Val(1))
    v.vxinter   .= trapz(kys,v.buffer[:,:,3],Val(1))
    v.vx        .= v.vxinter .+ v.vxintra
    v.vyintra   .= trapz(kys,v.buffer[:,:,5],Val(1))
    v.vyinter   .= trapz(kys,v.buffer[:,:,6],Val(1))
    v.vy        .= v.vyinter .+ v.vyintra
    return v
end


struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
    buffer::Matrix{T}
end
function Occupation(h::Hamiltonian{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0),Matrix{T}(undef,0,0))
end
function Occupation(p::NumericalParameters{T}) where {T<:Real}
    pars    = getparams(p)
    nt      = pars.nt
    nky     = pars.nky
    cbocc   = zeros(T,nt)
    buffer  = zeros(T,nky,nt)
    return Occupation(cbocc,buffer)
end

function resize(o::Occupation{T},p::NumericalParameters{T}) where {T<:Real}
    return Occupation(p)
end

getnames_obs(occ::Occupation{T}) where {T<:Real} = ["cbocc"]
getparams(occ::Occupation{T}) where {T<:Real}    = getnames_obs(occ)
getshortname(occ::Occupation{T}) where {T<:Real} = "Occupation"
arekresolved(occ::Occupation{T}) where {T<:Real} = [false, true]

@inline function addto!(o::Occupation{T},ototal::Occupation{T}) where {T<:Real}
    ototal.cbocc .= ototal.cbocc .+ o.cbocc
end

@inline function normalize!(o::Occupation{T},norm::T) where {T<:Real}
    o.cbocc ./= norm
end

function zero(o::Occupation{T}) where {T<:Real}
    cbocc = zero(o.cbocc)
    buffer = zero(o.buffer)
    return Occupation(cbocc,buffer)
end

function calcobs_k1d!(sim::Simulation{T},occ::Occupation{T},sol,ky::T) where {T<:Real}

end

function integrate1d_obs!(sim::Simulation{T},o::Occupation{T},sol,ky::T,ky_index::Integer,
                    moving_bz::Array{T,N}) where {T<:Real,N}

    p           = getparams(sim)
    kxs         = collect(p.kxsamples)
    
    calcobs_k1d!(sim,o,sol,ky)

    o.buffer[ky_index,:] .= trapz(kxs,sol[1:p.nkx,:] .* moving_bz)
end

function integrate2d_obs!(s::Simulation{T},occ::Occupation{T}) where {T<:Real}
    kys     = getparams(s).kysamples
    cbocc  = trapz(kys,o.buffer,Val(1))
end

function finalize_obs1d!(s::Simulation{T},o::Occupation{T}) where {T<:Real}
    
end

function calc_obs_k1d!(sim::Simulation{T},sol,ky::T,ky_index::Integer) where {T<:Real}

    p              = getparams(sim)
    ax             = get_vecpotx(sim.drivingfield)
    ay             = get_vecpoty(sim.drivingfield)
    moving_bz      = zeros(T,p.nkx,length(sol.t))

    sig(x)         = 0.5*(1.0+tanh(x/2.0)) # = logistic function 1/(1+e^(-t)) 
    bzmask1d(kx)   = sig((kx-p.bz[1])/(2*p.dkx)) * sig((p.bz[2]-kx)/(2*p.dkx))

    for i in 1:length(sol.t)
        moving_bz[:,i] .= bzmask1d.(p.kxsamples .- ax(sol.t[i]))
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
    
    for o in sim.observables
        integrate1d_obs!(sim,o,sol,ky,ky_index,moving_bz)
    end
end

function finalize_obs1d!(s::Simulation{T}) where {T<:Real}
    for o in s.observables
        finalize_obs1d!(s,o)
    end
end

function integrate2d_obs!(s::Simulation)
    for o in s.observables
        integrate2d_obs!(s,o)
    end
end

function resize_obs!(s::Simulation)
    s.observables .= [resize(o,s.numericalparams) for o in s.observables]
end
