export CartesianKGrid1d
export CartesianKGrid2d
export CartesianKGrid2dStrips
export KGrid0d
export KGridEmpty

abstract type CartesianKGrid{T} <: KGrid{T} end

function symmetric_steprange(max::Real, step::Real)
	step < zero(step) && return symmetric_steprange(max, -step)
	max < zero(max) && return symmetric_steprange(-max, step)
	hi = zero(step):step:max
	lo = zero(step):-step:-max
	return append!(reverse(collect(lo))[1:end-1], collect(hi))
end

"Empty k grid for initialization purposes"
struct KGridEmpty{T} <: KGrid{T} end

"""
	KGrid0d{T}(kx,ky) <: CartesianKGrid{T}

Zero-dimensional grid, i.e. a single point in k-space.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`CartesianKGrid2d`](@ref)
"""
struct KGrid0d{T <: Real} <: CartesianKGrid{T}
    kx::T
    ky::T
end

"""
	CartesianKGrid1d{T}(dkx, kxmax[, ky]) <: CartesianKGrid{T}

One-dimensional equidistant samples in k-space in kx direction at `ky`.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`KGrid0d`](@ref),
[`CartesianKGrid2d`](@ref)
"""
struct CartesianKGrid1d{T <: Real} <: CartesianKGrid{T}
    dkx::T
    kxmax::T
    ky::T
end

function CartesianKGrid1d(dkx::Real,kxmax::Real,ky::Real=0.0) 
    return CartesianKGrid1d(promote(dkx,kxmax,ky)...)
end
CartesianKGrid1d(kgrid::Dict) = construct_type_from_dict(CartesianKGrid1d, kgrid)

"""
	CartesianKGrid2d{T}(dkx, kxmax, dky, kymax) <: CartesianKGrid{T}

Two-dimensional equidistant cartesian grid in reciprocal (k-)space.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`KGrid0d`](@ref)
"""
struct CartesianKGrid2d{T <: Real} <: CartesianKGrid{T}
    dkx::T
    kxmax::T
    dky::T
    kymax::T
end

function CartesianKGrid2d(dkx::Real,kxmax::Real,dky::Real,kymax::Real)
    return CartesianKGrid2d(promote(dkx,kxmax,dky,kymax)...)
end

CartesianKGrid2d(kgrid::Dict) = construct_type_from_dict(CartesianKGrid2d, kgrid)


struct CartesianKGrid2dStrips{T <: Real} <: CartesianKGrid{T}
    dkx::T
    kxmax::T
    dky::T
    kymax::T
    kymin::T
end

function CartesianKGrid2d(kgrid::CartesianKGrid2dStrips)
    return CartesianKGrid2d(kgrid.dkx,kgrid.kxmax,kgrid.dky,kgrid.kymax)
end

function getkxsamples(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,CartesianKGrid2dStrips}) 
    return symmetric_steprange(kgrid.kxmax,kgrid.dkx)
end
getkxsamples(kgrid::KGrid0d) = [kgrid.kx]

function getkysamples(kgrid::CartesianKGrid2dStrips)
    fullsamples = symmetric_steprange(kgrid.kymax,kgrid.dky)
    return fullsamples[abs.(fullsamples) .> kgrid.kymin]
end
getkysamples(kgrid::CartesianKGrid2d) = symmetric_steprange(kgrid.kymax,kgrid.dky)
getkysamples(kgrid::Union{KGrid0d,CartesianKGrid1d}) = [kgrid.ky]

getdimension(::KGrid0d)                                         = UInt8(0)
getdimension(::CartesianKGrid1d)                                = UInt8(1)
getdimension(k::Union{CartesianKGrid2d,CartesianKGrid2dStrips}) = UInt8(2)


function getnkx(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,KGrid0d,CartesianKGrid2dStrips})
    return length(getkxsamples(kgrid))
end
function getnky(kgrid::Union{CartesianKGrid1d,CartesianKGrid2d,KGrid0d,CartesianKGrid2dStrips})
    return length(getkysamples(kgrid))
end

for func ∈ (:getkxsamples,:getnkx,:getkysamples,:getnky)
    @eval(Damysos,$func(s::Simulation) = $func(s.grid))
    @eval(Damysos,$func(g::NGrid)       = $func(g.kgrid))
end

function applyweights_afterintegration!(obs::Vector{<:Observable}, kgrid::CartesianKGrid)
    normalize!.(obs,1 / volume_element(kgrid))
end

volume_element(::KGrid0d)                       = 1
volume_element(kgrid::CartesianKGrid1d)         = kgrid.dkx
volume_element(kgrid::Union{CartesianKGrid2d,CartesianKGrid2dStrips}) = kgrid.dkx * kgrid.dky

function printparamsSI(kgrid::CartesianKGrid, us::UnitScaling; digits = 3)

    symbols     = fieldnames(typeof(kgrid))
    valuesSI    = [wavenumberSI(getproperty(kgrid,s),us) for s in symbols]
    values      = [getproperty(kgrid,s) for s in symbols]
    str         = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
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

getnk(kgrid::CartesianKGrid) = getnkx(kgrid) * getnky(kgrid)

function buildkgrid_chunks(kgrid::CartesianKGrid, kchunksize::Integer)
	kxs = collect(getkxsamples(kgrid))
	kys = collect(getkysamples(kgrid))
	ks  = [getkgrid_point(i, kxs, kys) for i in 1:getnk(kgrid)]
	return subdivide_vector(ks, kchunksize)
end
