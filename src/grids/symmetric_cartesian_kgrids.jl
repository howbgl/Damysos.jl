export CartesianKGrid1d
export CartesianKGrid2d
export KGrid0d

function symmetric_steprange(max::Real, step::Real)
	step < zero(step) && return symmetric_steprange(max, -step)
	max < zero(max) && return symmetric_steprange(-max, step)
	hi = zero(step):step:max
	lo = zero(step):-step:-max
	return append!(reverse(collect(lo))[1:end-1], collect(hi))
end

"""
	KGrid0d{T}(kx,ky) <: KGrid{T}

Zero-dimensional grid, i.e. a single point in k-space.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`CartesianKGrid2d`](@ref)
"""
struct KGrid0d{T <: Real} <: KGrid{T}
    kx::T
    ky::T
end

"""
	CartesianKGrid1d{T}(dkx, kxmax[, ky]) <: KGrid{T}

One-dimensional equidistant samples in k-space in kx direction at `ky`.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`KGrid0d`](@ref),
[`CartesianKGrid2d`](@ref)
"""
struct CartesianKGrid1d{T <: Real} <: KGrid{T} 
    dkx::T
    kxmax::T
    ky::T
end

function CartesianKGrid1d(dkx::Real,kxmax::Real,ky::Real=0.0) 
    return CartesianKGrid1d(promote(dkx,kxmax,ky)...)
end
CartesianKGrid1d(kgrid::Dict) = construct_type_from_dict(CartesianKGrid1d, kgrid)

"""
	CartesianKGrid2d{T}(dkx, kxmax, dky, kymax) <: KGrid{T} 

Two-dimensional equidistant cartesian grid in reciprocal (k-)space.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`KGrid0d`](@ref)
"""
struct CartesianKGrid2d{T <: Real} <: KGrid{T} 
    dkx::T
    kxmax::T
    dky::T
    kymax::T
end

function CartesianKGrid2d(dkx::Real,kxmax::Real,dky::Real,kymax::Real)
    return CartesianKGrid2d(promote(dkx,kxmax,dky,kymax)...)
end

CartesianKGrid2d(kgrid::Dict) = construct_type_from_dict(CartesianKGrid2d, kgrid)


function getkxsamples(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d}) 
    return symmetric_steprange(kgrid.kxmax,kgrid.dkx)
end
getkxsamples(kgrid::KGrid0d) = [kgrid.kx]

getkysamples(kgrid::CartesianKGrid2d) = symmetric_steprange(kgrid.kymax,kgrid.dky)
getkysamples(kgrid::Union{KGrid0d,CartesianKGrid1d}) = [kgrid.ky]

getdimension(kgrid::KGrid0d)            = UInt8(0)
getdimension(::CartesianKGrid1d)        = UInt8(1)
getdimension(kgrid::CartesianKGrid2d)   = UInt8(2)


function getnkx(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,KGrid0d})
    return length(getkxsamples(kgrid))
end
function getnky(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,KGrid0d})
    return length(getkysamples(kgrid))
end

for func ∈ (:getkxsamples,:getnkx,:getkysamples,:getnky)
    @eval(Damysos,$func(s::Simulation) = $func(s.grid))
    @eval(Damysos,$func(g::NGrid)       = $func(g.kgrid))
end


function printparamsSI(kgrid::KGrid0d, us::UnitScaling; digits = 3)
    params  = [("kx",kgrid.dkx), ("ky",kgrid.kxmax)]
    strings = [param_string(p[1],wavenumberSI(p[2], us),p[2]; digits = digits) for p in params]
    return strings[1] * "\n" * strings[2] *"\n" 
end

function printparamsSI(kgrid::CartesianKGrid1d, us::UnitScaling; digits = 3)
    params  = [("dkx",kgrid.dkx), ("kxmax",kgrid.kxmax)]
    strings = [param_string(p[1],wavenumberSI(p[2], us),p[2]; digits = digits) for p in params]
    return strings[1] * "\n" * strings[2] *"\n" 
end

function printparamsSI(kgrid::CartesianKGrid2d, us::UnitScaling; digits = 3)
    params  = [
        ("dkx",kgrid.dkx), 
        ("kxmax",kgrid.kxmax),
        ("dky",kgrid.dky), 
        ("kymax",kgrid.kymax)]
    strings = [param_string(p[1],wavenumberSI(p[2], us),p[2]; digits = digits) for p in params]
    return strings[1] * "\n" * strings[2] *"\n" 
end


function Base.show(
    io::IO,
    ::MIME"text/plain",
    kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,KGrid0d})

    if kgrid isa CartesianKGrid1d
        println(io, "CartesianKGrid1d:")
    elseif kgrid isa CartesianKGrid2d
        println(io, "CartesianKGrid2d:")
    else
        println(io, "KGrid0d:")
    end
    print(io, prepend_spaces(printfields_generic(kgrid)))
end


@inline cartesianindex2dx(i,n) = 1 + ((i-1) % n)
@inline cartesianindex2dy(i,n) = 1 + ((i-1) ÷ n)
@inline caresianindex2d(i,n)   = (cartesianindex2dx(i,n),cartesianindex2dy(i,n))


@inline function get_cartesianindices_kgrid(
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    return CartesianIndices((length(kxsamples),length(kysamples)))
end

@inline function getkgrid_index(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    return get_cartesianindices_kgrid(kxsamples,kysamples)[i]
end

@inline function getkgrid_index(i::Integer,nkx::Integer,nky::Integer)
    return caresianindex2d(i,nkx)
end

@inline function getkgrid_point(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = caresianindex2d(i,length(kxsamples))

    return SA[kxsamples[idx[1]],kysamples[idx[2]]]
end


@inline function getkgrid_point_kx(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = getkgrid_index(i,kxsamples,kysamples)

    return kxsamples[idx[1]]
end
@inline function getkgrid_point_ky(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = getkgrid_index(i,kxsamples,kysamples)

    return kysamples[idx[2]]
end

