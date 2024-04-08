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

function resize(o::Occupation,p::NumericalParameters)
    return Occupation(p)
end

function empty(o::Occupation)
    return Occupation(o)
end

getnames_obs(occ::Occupation)   = ["cbocc", "cbocck"]
getparams(occ::Occupation)      = getnames_obs(occ)
arekresolved(occ::Occupation)   = [false, true]

@inline function addto!(o::Occupation,ototal::Occupation)
    ototal.cbocc .= ototal.cbocc .+ o.cbocc
end

@inline function copyto!(odest::Occupation,osrc::Occupation)
    odest.cbocc .= osrc.cbocc
end

@inline function normalize!(o::Occupation,norm::Real)
    o.cbocc ./= norm
end

+(o1::Occupation,o2::Occupation) = Occupation(o1.cbocc .+ o2.cbocc)
-(o1::Occupation,o2::Occupation) = Occupation(o1.cbocc .- o2.cbocc)
*(o::Occupation,x::Number)       = Occupation(x .* o.cbocc)
*(x::Number,o::Occupation)       = o*x
zero(o::Occupation)              = Occupation(zero(o.cbocc))

function buildobservable_expression_upt(sim::Simulation,::Occupation)
    return :(real(u[1]))
end

function buildobservable_expression(sim::Simulation,o::Occupation) 
    return :(real(u[1]))
end


function write_ensembledata_to_observable!(o::Occupation,data::Vector{<:Real})

    length(o.cbocc) != length(data) && throw(ArgumentError(
        """
        data must be same length as observable Occupation. Got lengths of \
        $(length(data)) and $(length(o.cbocc))"""))

    o.cbocc .= data
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
