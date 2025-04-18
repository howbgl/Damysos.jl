export Occupation
"""
    Occupation{T<:Real} <: Observable{T}

Holds time series data of the occupation computed from the density matrix.

Only the conduction band occupation ``\\rho_{cc}(t)`` is stored since ``Tr\\rho(t)=1``


# Fields
- `cbocc::Vector{T}`: time-dependent conduction band occupation ``\\rho_{cc}(t)``

# See also
[`Velocity`](@ref Velocity)
"""
struct Occupation{T<:Real} <: Observable{T}
    cbocc::Vector{T}
end
function Occupation(::SimulationComponent{T}) where {T<:Real}
    return Occupation(Vector{T}(undef,0))
end
function Occupation(g::NGrid{T}) where {T<:Real}
    return Occupation(zeros(T,getnt(g)))
end

function resize(::Occupation,g::NGrid)
    return Occupation(g)
end
function resize(::Occupation{T},nt::Integer) where {T<:Real}
    return Occupation(zeros(T,nt))
end

function Base.append!(o1::Occupation,o2::Occupation)
    append!(o1.cbocc,o2.cbocc)
    return o1
end

function empty(o::Occupation)
    return Occupation(o)
end

getnames_obs(occ::Occupation)   = ["cbocc", "cbocck"]
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

function Base.isapprox(
    o1::Occupation{T},
    o2::Occupation{U};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    cb1 = deepcopy(o1.cbocc)
    cb2 = deepcopy(o2.cbocc)
    upsample!(cb1,cb2)

    return Base.isapprox(cb1,cb2;atol=atol,rtol=rtol,nans=nans)
end

buildobservable_expression_svec_upt(sim::Simulation,::Occupation) = :(SA[real(u[1])])
buildobservable_vec_of_expr(sim::Simulation,::Occupation)         = [:(real(u[1]))]


function sum_observables!(
    o::Occupation,
    funcs::Vector,
    d_kchunk::CuArray{<:SVector{2,<:Real}},
    d_us::CuArray{<:SVector{2,<:Complex}},
    d_ts::CuArray{<:Real,2},
    d_weigths::CuArray{<:Real,2},
    buf::CuArray{<:Real,2})

    cb          = funcs[1]
    buf         .= cb.(d_us,d_kchunk',d_ts) .* d_weigths
    total       = reduce(+,buf;dims=2)
    o.cbocc     .= Array(total)
    return o
end

function calculate_observable_singlemode!(sim::Simulation,o::Occupation,f,res::ODESolution)
    cbocc(u,t) = f[1](u,res.prob.p,t)
    o.cbocc .= cbocc.(res.u,res.t)
    return nothing
end

function write_ensembledata_to_observable!(o::Occupation,data::Vector{<:Real})

    length(o.cbocc) != length(data) && throw(ArgumentError(
        """
        data must be same length as observable Occupation. Got lengths of \
        $(length(data)) and $(length(o.cbocc))"""))

    o.cbocc .= data
end

function write_ensembledata_to_observable!(
    o::Occupation,
    data::Vector{<:SVector{1,<:Real}})

    length(o.cbocc) != length(data) && throw(ArgumentError(
        """
        data must be same length as observable Occupation. Got lengths of \
        $(length(data)) and $(length(o.cbocc))"""))

    for (i,oi) in enumerate(data)
        o.cbocc[i] = oi[1]
    end
end
