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
- `vyinter::Vector{T}`: ``\\rho_{cv}(t) v^{y}_{vc}(t)  + \\rho_{vc}(t) v^{y}(t) _{cv}(t) ``

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
function Velocity(h::Hamiltonian{T}) where {T<:Real}
    return Velocity(Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0),
                    Vector{T}(undef,0),Vector{T}(undef,0),Vector{T}(undef,0))
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

function resize(v::Velocity,p::NumericalParameters)
    return Velocity(p)
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

function buildobservable_expression_upt(sim::Simulation,::Velocity)
    
    h    = sim.liouvillian.hamiltonian
    df   = sim.drivingfield

    vxvc = vx_vc(h)
    vxcc = vx_cc(h)
    vxvv = vx_vv(h)
    vyvc = vy_vc(h)
    vycc = vy_cc(h)
    vyvv = vy_vv(h)

    ax  = vecpotx(df)
    ay  = vecpoty(df)
    
    vxintra_expr = :(real(cc) * ($vxcc - $vxvv))
    vxinter_expr = :(2real(cv * $vxvc))
    vyintra_expr = :(real(cc) * ($vycc - $vyvv))
    vyinter_expr = :(2real(cv * $vyvc))
    vel_expr     = :(SA[$vxintra_expr,$vxinter_expr,$vyintra_expr,$vyinter_expr])

    replace_expression!(vel_expr,:kx,:(kx-$ax))
    replace_expression!(vel_expr,:cc,:(u[1]))
    replace_expression!(vel_expr,:cv,:(u[2]))
    replace_expression!(vel_expr,:kx,:(p[1]))
    replace_expression!(vel_expr,:ky,:(p[2]))
    
    return vel_expr
end

function buildobservable_expression(sim::Simulation,v::Velocity)
    
    h    = sim.liouvillian.hamiltonian
    df   = sim.drivingfield

    vxvc = vx_vc(h)
    vxcc = vx_cc(h)
    vxvv = vx_vv(h)
    vyvc = vy_vc(h)
    vycc = vy_cc(h)
    vyvv = vy_vv(h)

    ax  = vecpotx(df)
    ay  = vecpoty(df)
    
    vxintra_expr = :(real(u[1]) * ($vxcc - $vxvv))
    vxinter_expr = :(2real(u[2] * $vxvc))
    vyintra_expr = :(real(u[1]) * ($vycc - $vyvv))
    vyinter_expr = :(2real(u[2] * $vyvc))
    vel_expr     = :(SA[$vxintra_expr,$vxinter_expr,$vyintra_expr,$vyinter_expr])

    replace_expression!(vel_expr,:kx,:(kx-$ax))
    return vel_expr
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
    
    for i in 1:4
        v.vxintra[timeindex] = data[1]
        v.vxinter[timeindex] = data[2]
        v.vyintra[timeindex] = data[3]
        v.vyinter[timeindex] = data[4]
        v.vx[timeindex] = v.vxinter[timeindex] + v.vxintra[timeindex]
        v.vy[timeindex] = v.vyinter[timeindex] + v.vyintra[timeindex]
    end
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
