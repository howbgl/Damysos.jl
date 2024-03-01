export Occupation
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

+(o1::Occupation,o2::Occupation) = Occupation(o1.cbocc .+ o2.cbocc)
-(o1::Occupation,o2::Occupation) = Occupation(o1.cbocc .- o2.cbocc)
*(o::Occupation,x::Number)       = Occupation(x .* o.cbocc)
*(x::Number,o::Occupation)       = o*x
zero(o::Occupation)              = Occupation(zero(o.cbocc))

function isapprox(
    o1::Occupation{T},
    o2::Occupation{U};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    cb1 = deepcopy(o1.cbocc)
    cb2 = deepcopy(o2.cbocc)
    upsample!(cb1,cb2)

    return isapprox(cb1,cb2;atol=atol,rtol=rtol,nans=nans)
end


function getfuncs(sim::Simulation,o::Occupation)
    return []
end


function integrateobs_kxbatch_add!(
    sim::Simulation{T},
    o::Occupation{T},
    sol,
    kxsamples::AbstractVector{T},
    ky::T,
    moving_bz::AbstractMatrix{T},
    funcs) where {T<:Real}

    ts    = getparams(sim).tsamples
    nkx   = length(kxsamples)

    for i in eachindex(ts)
        ρcc             = @view sol[1:nkx,i]
        o.cbocc[i]      += trapz(kxsamples,moving_bz[:,i] .* real.(ρcc))
    end
    return o
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


function integrateobs!(
    occs::Vector{Occupation{T}},
    odest::Occupation{T},
    vertices::Vector{T}) where {T<:Real}

    odest.cbocc .= trapz((:,hcat(vertices)),hcat([o.cbocc for o in occs]...))
end
