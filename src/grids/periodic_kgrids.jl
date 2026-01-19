export HexagonalMPKGrid2d
export CartesianMPKGrid1d

abstract type HexagonalKGrid2d{T} <: PeriodicKGrid{T} end

function reciprocal_primitive_vectors(::Type{T}, a) where {T<:HexagonalKGrid2d}
    b1 = (2π / a) * SVector(1/√3, 1)
    b2 = (2π / a) * SVector(1/√3, -1)
    return b1, b2   
end

function u_monkhorst_pack(r::Integer, q::Integer)
    return (2r-q-1) / 2q
end


"""
	HexagonalMPKGrid2d{T}(dk1, dk2, a) <: HexagonalKGrid2d{T}

Two-dimensional Monkhorst-Pack mesh for hexagonal grid in reciprocal (k-)space.

# Reference
<https://doi.org/10.1103/PhysRevB.13.5188>

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`KGrid0d`](@ref)
"""
struct HexagonalMPKGrid2d{T <: Real} <: HexagonalKGrid2d{T}
    dk1::T
    dk2::T
    a::T
    function HexagonalMPKGrid2d{T}(dk1::T, dk2::T, a::T) where {T <: Real}
        @argcheck dk1 > 0
        @argcheck dk2 > 0
        @argcheck a > 0

		b1, b2 = reciprocal_primitive_vectors(HexagonalMPKGrid2d, a)
        q1 = round(Int, norm(b1) / dk1)
        q2 = round(Int, norm(b2) / dk2)
        if q1<2 || q2<2
            throw(ArgumentError(
                "HexagonalMPKGrid2d requires at least 2 k-points in each direction. Got $(q1), $(q2)."))            
        end
		return new(norm(b1) / q1, norm(b2) / q2, a)
	end
end
HexagonalMPKGrid2d(dk1::T,dk2::T,a::T) where {T <: Real} = HexagonalMPKGrid2d{T}(dk1,dk2,a)
HexagonalMPKGrid2d(dk1,dk2,a) = HexagonalMPKGrid2d(promote(dk1,dk2,a)...)

getdimension(::HexagonalMPKGrid2d) = UInt8(2)

function applyweights_afterintegration!(obs::Vector{<:Observable}, kgrid::HexagonalMPKGrid2d)
    normalize!.(obs,1 / volume_element(kgrid))
end

volume_element(kgrid::HexagonalMPKGrid2d) =  kgrid.dk1 * kgrid.dk2

function reciprocal_primitive_vectors(kgrid::HexagonalMPKGrid2d)
    return reciprocal_primitive_vectors(HexagonalMPKGrid2d, kgrid.a)
end

function getqs(kgrid::HexagonalMPKGrid2d)
    b1, b2 = reciprocal_primitive_vectors(kgrid)
    q1 = round(Int, norm(b1) / kgrid.dk1)
    q2 = round(Int, norm(b2) / kgrid.dk2)
    return q1, q2
end

function getksamples(kgrid::HexagonalMPKGrid2d)
	b1, b2 = reciprocal_primitive_vectors(kgrid)
    q1, q2 = getqs(kgrid)
    inds   = CartesianIndices((q1,q2))
    kmat   = [u_monkhorst_pack(I[1],q1)*b1 + u_monkhorst_pack(I[2],q2)*b2 for I in inds]
    return vec(kmat)
end

ntrajectories(kgrid::HexagonalMPKGrid2d) = reduce(*, getqs(kgrid))

function buildkgrid_chunks(kgrid::HexagonalMPKGrid2d, kchunksize::Integer)
	return subdivide_vector(getksamples(kgrid), kchunksize)
end

function printparamsSI(kgrid::HexagonalMPKGrid2d, us::UnitScaling; digits = 3)

    symbols     = ["dk1","dk2","a", "nkoints"]
    values      = [kgrid.dk1, kgrid.dk2, kgrid.a, ntrajectories(kgrid)]
    valuesSI    = [wavenumberSI(kgrid.dk1,us), wavenumberSI(kgrid.dk2,us), lengthSI(kgrid.a,us), ntrajectories(kgrid)]
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
    kgrid::HexagonalMPKGrid2d)

    println(io, "HexagonalMPKGrid2d:")
    print(io, prepend_spaces(printfields_generic(kgrid)))
end

struct CartesianMPKGrid1d{T <: Real} <: PeriodicKGrid{T} 
    dkx::T
    a::T
    function CartesianMPKGrid1d{T}(dkx::T, a::T) where {T <: Real}
        @argcheck dkx > 0
        @argcheck a > 0

        q1    = round(Int, 2π / dkx)
        if q1<2
            throw(ArgumentError(
                "CartesianMPKGrid1d requires at least 2 k-points. Got $(q1)."))            
        end
        return new(2π / q1, a)
    end
end
CartesianMPKGrid1d(dkx::T,a::T) where {T <: Real} = CartesianMPKGrid1d{T}(dkx,a)
CartesianMPKGrid1d(dkx,a) = CartesianMPKGrid1d(promote(dkx,a)...)

getdimension(::CartesianMPKGrid1d) = UInt8(1)

function applyweights_afterintegration!(obs::Vector{<:Observable}, kgrid::CartesianMPKGrid1d)
    normalize!.(obs,1 / volume_element(kgrid))
end

volume_element(kgrid::CartesianMPKGrid1d) =  kgrid.dkx

getqs(kgrid::CartesianMPKGrid1d) = round(Int, 2π / kgrid.dkx)

ntrajectories(kgrid::CartesianMPKGrid1d) = getqs(kgrid)

function getksamples(kgrid::CartesianMPKGrid1d)
    q1 = round(Int, 2π / kgrid.dkx)
    kxs = [u_monkhorst_pack(I, q1) * (2π / q1) for I in 1:q1]
    return [SA[kx,zero(kx)] for kx in kxs]
end

function buildkgrid_chunks(kgrid::CartesianMPKGrid1d, kchunksize::Integer)
    return subdivide_vector(getksamples(kgrid), kchunksize)
end

function printparamsSI(kgrid::CartesianMPKGrid1d, us::UnitScaling; digits = 3)

    symbols     = ["dkx","a", "nkoints"]
    values      = [kgrid.dkx, kgrid.a, ntrajectories(kgrid)]
    valuesSI    = [wavenumberSI(kgrid.dkx,us), lengthSI(kgrid.a,us), ntrajectories(kgrid)]
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
    kgrid::CartesianMPKGrid1d)

    println(io, "CartesianMPKGrid1d:")
    print(io, prepend_spaces(printfields_generic(kgrid)))
end