export VelocityX
"""
    VelocityX{T<:Real} <: Observable{T}

Holds time series data of the physical velocity in x direction.

The velocity is computed via
```math
\\vec{v}(t) = Tr [\\rho(t) \\frac{\\partial H}{\\partial\\vec{k}}]
            = \\rho_{cc}(t) \\vec{v}_{cc}(t)  + \\rho_{cv}(t) \\vec{v}_{vc}(t) 
            + \\rho_{vv}(t) \\vec{v}_{vv}(t)  + \\rho_{vc}(t) \\vec{v}_{cv}(t) 
``` 


# Fields
- `vx::Vector{T}`: total velocity in x-direction. 
- `vxintra::Vector{T}`: ``\\rho_{cc}(t) v^{x}_{cc}(t)  + \\rho_{vv}(t) v^{x}_{vv}(t) ``
- `vxinter::Vector{T}`: ``\\rho_{cv}(t) v^{x}_{vc}(t)  + \\rho_{vc}(t) v^{x}_{cv}(t) ``

# See also
[`Occupation`](@ref Occupation)
[`Velocity`](@ref Velocity)
"""
struct VelocityX{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
end
function VelocityX(::SimulationComponent{T}) where {T<:Real}
    return VelocityX(Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0))
end

# TODO make this file more concise, atm it is copy & paste from Velocity.jl

function VelocityX(g::NGrid{T}) where {T<:Real}

    nt          = getnt(g)
    vxintra     = zeros(T,nt)
    vxinter     = zeros(T,nt)
    vx          = zeros(T,nt)

    return VelocityX(
        vx,
        vxintra,
        vxinter)
end

function VelocityX(
    vxintra::Vector{T},
    vxinter::Vector{T}) where {T<:Real}
    return VelocityX(vxintra .+ vxinter,vxintra,vxinter)
end

resize(v::VelocityX,g::NGrid) = VelocityX(g)

function resize(::VelocityX{T},nt::Integer) where {T<:Real}
    return VelocityX(zeros(T,nt),zeros(T,nt),zeros(T,nt))
end

function Base.append!(v1::VelocityX,v2::VelocityX)
    append!(v1.vx,v2.vx)
    append!(v1.vxintra,v2.vxintra)
    append!(v1.vxinter,v2.vxinter)
    return v1
end

function empty(v::VelocityX) 
    return VelocityX(v)
end


getnames_obs(v::VelocityX)   = ["vx","vxintra","vxinter"]
arekresolved(v::VelocityX)   = [false,false,false]


@inline function addto!(v::VelocityX,vtotal::VelocityX)
    vtotal.vx .= vtotal.vx .+ v.vx
    vtotal.vxinter .= vtotal.vxinter .+ v.vxinter
    vtotal.vxintra .= vtotal.vxintra .+ v.vxintra
end

@inline function copyto!(vdest::VelocityX,vsrc::VelocityX)
    vdest.vx        .= vsrc.vx
    vdest.vxintra   .= vsrc.vxintra
    vdest.vxinter   .= vsrc.vxinter
end

@inline function normalize!(v::VelocityX,norm::Real)
    v.vx ./= norm
    v.vxinter ./= norm
    v.vxintra ./= norm
end

function +(v1::VelocityX,v2::VelocityX)
    VelocityX(
        v1.vx .+ v2.vx,
        v1.vxintra .+ v2.vxintra,
        v1.vxinter .+ v2.vxinter)
end
function -(v1::VelocityX,v2::VelocityX)
    VelocityX(
        v1.vx .- v2.vx,
        v1.vxintra .- v2.vxintra,
        v1.vxinter .- v2.vxinter)
end
*(v::VelocityX,x::Number) = VelocityX(
    x .* v.vx,
    x .* v.vxintra,
    x .* v.vxinter)
*(x::Number,v::VelocityX) = v * x

function zero(v::VelocityX) 
    VelocityX(
        zero(v.vx),
        zero(v.vxintra),
        zero(v.vxinter))
end


function Base.isapprox(
    v1::VelocityX{T},
    v2::VelocityX{U};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    vx1 = deepcopy(v1.vx)
    vx2 = deepcopy(v2.vx)

    upsample!(vx1,vx2)

    return isapprox(vx1,vx2;atol=atol,rtol=rtol,nans=nans)
end

function build_expression_velocity_svec(h::Hamiltonian,::VelocityX)

    vxintra_expr = build_expression_vxintra(h)
    vxinter_expr = build_expression_vxinter(h)

    return :(SA[$vxintra_expr,$vxinter_expr])
end

function buildobservable_expression_svec_upt(sim::Simulation,v::VelocityX)
    
    h   = sim.liouvillian.hamiltonian
    df  = sim.drivingfield
    ax  = vecpotx(df)
    ay  = vecpoty(df)
    
    vel_expr = build_expression_velocity_svec(h,v)
    rules    = Dict(
        :kx => :(p[1] - $ax),
        :ky => :(p[2]),
        :cc => :(u[1]),
        :cv => :(u[2]))
    
    return replace_expressions!(vel_expr,rules)
end

function buildobservable_vec_of_expr(sim::Simulation,::VelocityX)

    h   = sim.liouvillian.hamiltonian
    df  = sim.drivingfield
    ax  = vecpotx(df)
    ay  = vecpoty(df)

    vxintra = build_expression_vxintra(h)
    vxinter = build_expression_vxinter(h)
    rules   = Dict(
        :kx => :(p[1] - $ax),
        :ky => :(p[2]),
        :cc => :(u[1]),
        :cv => :(u[2]))
    
    for v in (vxintra,vxinter)
        replace_expressions!(v,rules)
    end
    
    return [vxintra,vxinter]
end

function sum_observables!(
    v::VelocityX,
    funcs::Vector,
    d_kchunk::CuArray{<:SVector{2,<:Real}},
    d_us::CuArray{<:SVector{2,<:Complex}},
    d_ts::CuArray{<:Real,2},
    d_weigths::CuArray{<:Real,2},
    buf::CuArray{<:Real,2})

    vcontributions = (v.vxintra,v.vxinter)

    for (vm,f) in zip(vcontributions,funcs) 
        buf     .= f.(d_us,d_kchunk',d_ts) .* d_weigths
        total   = reduce(+,buf;dims=2)
        vm      .= Array(total)
    end
    v.vx .= v.vxintra .+ v.vxinter
    return v
end

function calculate_observable_singlemode!(sim::Simulation,v::VelocityX,f,res::ODESolution)
    funcs = [(u,t) -> func(u,res.prob.p,t) for func in f]
    vs    = (v.vxintra,v.vxinter)
    for (vv,ff) in zip(vs,funcs)
        vv .= ff.(res.u,res.t)
    end
    
    v.vx .= v.vxintra .+ v.vxinter

    return nothing
end


function write_ensembledata_to_observable!(v::VelocityX,data::Vector{<:SVector{2,<:Real}})

    length(v.vx) != length(data) && throw(ArgumentError(
        """
        data must be same length as observable Velocity. Got lengths of \
        $(length(data)) and $(length(v.vx))"""))

    for (i,d) in enumerate(data)
        write_svec_timeslice_to_observable!(v,i,d)
    end
end

function write_svec_timeslice_to_observable!(
    v::VelocityX,
    timeindex::Integer,
    data::SVector{2,<:Real})
    
    v.vxintra[timeindex] = data[1]
    v.vxinter[timeindex] = data[2]
    v.vx[timeindex] = v.vxinter[timeindex] + v.vxintra[timeindex]
end

function getfuncs(sim::Simulation,v::VelocityX)
    df = sim.drivingfield
    h  = sim.hamiltonian
    return [get_vecpotx(df),get_vecpoty(df),getvx_cc(h),getvx_vc(h),getvx_vv(h)]
end
