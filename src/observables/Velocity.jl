export Velocity
"""
    Velocity{T<:Real} <: Observable{T}

Holds time series data of the physical velocity computed from the density matrix.

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
- `vy::Vector{T}`: total velocity in y-direction. 
- `vyintra::Vector{T}`: ``\\rho_{cc}(t) v^{y}_{cc}(t)  + \\rho_{vv}(t) v^{y}_{vv}(t) ``
- `vyinter::Vector{T}`: ``\\rho_{cv}(t) v^{y}_{vc}(t)  + \\rho_{vc}(t) v^{y}_{cv}(t) ``

# See also
[`Occupation`](@ref Occupation)
"""
struct Velocity{T<:Real} <: Observable{T}
    vx::Vector{T}
    vxintra::Vector{T}
    vxinter::Vector{T}
    vy::Vector{T}
    vyintra::Vector{T}
    vyinter::Vector{T}
end
function Velocity(::SimulationComponent{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0),
                    Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0))
end

function Velocity(p::NumericalParameters{T}) where {T<:Real}

    nt          = getnt(p)

    vxintra     = zeros(T,nt)
    vxinter     = zeros(T,nt)
    vx          = zeros(T,nt)
    vyintra     = zeros(T,nt)
    vyinter     = zeros(T,nt)
    vy          = zeros(T,nt)

    return Velocity(
        vx,
        vxintra,
        vxinter,
        vy,
        vyintra,
        vyinter)
end

function Velocity(
    vxintra::Vector{T},
    vxinter::Vector{T},
    vyintra::Vector{T},
    vyinter::Vector{T}) where {T<:Real}
    return Velocity(vxintra .+ vxinter,vxintra,vxinter,vyintra .+ vyinter,vyintra,vyinter)
end

function resize(v::Velocity,p::NumericalParameters)
    return Velocity(p)
end

function resize(::Velocity{T},nt::Integer) where {T<:Real}
    return Velocity(zeros(T,nt),zeros(T,nt),zeros(T,nt),zeros(T,nt),zeros(T,nt),zeros(T,nt))
end

function Base.append!(v1::Velocity,v2::Velocity)
    append!(v1.vx,v2.vx)
    append!(v1.vxintra,v2.vxintra)
    append!(v1.vxinter,v2.vxinter)
    append!(v1.vy,v2.vy)
    append!(v1.vyintra,v2.vyintra)
    append!(v1.vyinter,v2.vyinter)
    return v1
end

function empty(v::Velocity) 
    return Velocity(v)
end


getnames_obs(v::Velocity)   = ["vx","vxintra","vxinter","vy","vyintra","vyinter"]
getparams(v::Velocity)      = getnames_obs(v)
arekresolved(v::Velocity)   = [false,false,false,false,false,false]


@inline function addto!(v::Velocity,vtotal::Velocity)
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

@inline function normalize!(v::Velocity,norm::Real)
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


function Base.isapprox(
    v1::Velocity{T},
    v2::Velocity{U};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    vx1 = deepcopy(v1.vx)
    vx2 = deepcopy(v2.vx)
    vy1 = deepcopy(v1.vy)
    vy2 = deepcopy(v2.vy)

    upsample!(vx1,vx2)
    upsample!(vy1,vy2)

    return all([
        isapprox(vx1,vx2;atol=atol,rtol=rtol,nans=nans),
        isapprox(vy1,vy2;atol=atol,rtol=rtol,nans=nans)])
end

build_expression_vxintra(h::Hamiltonian) = :(real(cc) * $(vx_cc(h)) + (1-real(cc)) * $(vx_vv(h)))
build_expression_vxinter(h::Hamiltonian) = :(2real(cv * $(vx_vc(h))))
build_expression_vyintra(h::Hamiltonian) = :(real(cc) * $(vy_cc(h)) + (1-real(cc)) * $(vy_vv(h)))
build_expression_vyinter(h::Hamiltonian) = :(2real(cv * $(vy_vc(h))))

function build_expression_velocity_svec(h::Hamiltonian,::Velocity)

    vxintra_expr = build_expression_vxintra(h)
    vxinter_expr = build_expression_vxinter(h)
    vyintra_expr = build_expression_vyintra(h)
    vyinter_expr = build_expression_vyinter(h)

    return :(SA[$vxintra_expr,$vxinter_expr,$vyintra_expr,$vyinter_expr])
end

function buildobservable_expression_svec_upt(sim::Simulation,v::Velocity)
    
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

function buildobservable_vec_of_expr(sim::Simulation,::Velocity)

    h   = sim.liouvillian.hamiltonian
    df  = sim.drivingfield
    ax  = vecpotx(df)
    ay  = vecpoty(df)

    vxintra = build_expression_vxintra(h)
    vxinter = build_expression_vxinter(h)
    vyintra = build_expression_vyintra(h)
    vyinter = build_expression_vyinter(h)
    rules   = Dict(
        :kx => :(p[1] - $ax),
        :ky => :(p[2]),
        :cc => :(u[1]),
        :cv => :(u[2]))
    
    for v in (vxintra,vxinter,vyintra,vyinter)
        replace_expressions!(v,rules)
    end
    
    return [vxintra,vxinter,vyintra,vyinter]
end

function sum_observables!(
    v::Velocity,
    funcs::Vector,
    d_kchunk::CuArray{<:SVector{2,<:Real}},
    d_us::CuArray{<:SVector{2,<:Complex}},
    d_ts::CuArray{<:Real,2},
    d_weigths::CuArray{<:Real,2},
    buf::CuArray{<:Real,2})

    vcontributions = (v.vxintra,v.vxinter,v.vyintra,v.vyinter)

    for (vm,f) in zip(vcontributions,funcs) 
        buf     .= f.(d_us,d_kchunk',d_ts) .* d_weigths
        total   = reduce(+,buf;dims=2)
        vm      .= Array(total)
    end
    v.vx .= v.vxintra .+ v.vxinter
    v.vy .= v.vyintra .+ v.vyinter
    return v
end

function calculate_observable_singlemode!(sim::Simulation,v::Velocity,f,res::ODESolution)
    funcs = [(u,t) -> func(u,res.prob.p,t) for func in f]
    vs    = (v.vxintra,v.vxinter,v.vyintra,v.vyinter)
    for (vv,ff) in zip(vs,funcs)
        vv .= ff.(res.u,res.t)
    end
    
    v.vx .= v.vxintra .+ v.vxinter
    v.vy .= v.vyintra .+ v.vyinter

    return nothing
end


function write_ensembledata_to_observable!(v::Velocity,data::Vector{<:SVector{4,<:Real}})

    length(v.vx) != length(data) && throw(ArgumentError(
        """
        data must be same length as observable Velocity. Got lengths of \
        $(length(data)) and $(length(v.vx))"""))

    for (i,d) in enumerate(data)
        write_svec_timeslice_to_observable!(v,i,d)
    end
end

function write_svec_timeslice_to_observable!(
    v::Velocity,
    timeindex::Integer,
    data::SVector{4,<:Real})
    
    v.vxintra[timeindex] = data[1]
    v.vxinter[timeindex] = data[2]
    v.vyintra[timeindex] = data[3]
    v.vyinter[timeindex] = data[4]
    v.vx[timeindex] = v.vxinter[timeindex] + v.vxintra[timeindex]
    v.vy[timeindex] = v.vyinter[timeindex] + v.vyintra[timeindex]
end

function getfuncs(sim::Simulation,v::Velocity)
    df = sim.drivingfield
    h  = sim.hamiltonian
    return [get_vecpotx(df),get_vecpoty(df),getvx_cc(h),getvx_vc(h),getvx_vv(h),
            getvy_cc(h),getvy_vc(h),getvy_vv(h)]
end

@inline function vintra(kx::Real,ky::Real,ρcc::Complex,vcc,vvv)
    return vintra(kx,ky,real(ρcc),vcc,vvv)
end
@inline function vintra(kx::T,ky::T,ρcc::T,vcc,vvv) where {T<:Real}
    return vcc(kx,ky)*ρcc + vvv(kx,ky)*(oneunit(T)-ρcc)
end

@inline function vinter(kx::Real,ky::Real,ρcv::Complex,vvc)
    return 2 * real(vvc(kx,ky) * ρcv)
end
