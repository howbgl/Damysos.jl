export NGrid

"""
	NGrid{T}(kgrid, tgrid)

Represents the discretzation of a [`Simulation`](@ref) in reciprocal (k-)space and time.

# Fields
- `kgrid::KGrid{T}`: set of points in k-space used for integration of observables.
- `tgrid::TimeGrid{T}`: points in time-space to evaluate observables.

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`CartesianKGrid2d`](@ref) [`KGrid0d`](@ref)
"""
struct NGrid{T <: Real} <: SimulationComponent{T}
    kgrid::KGrid{T}
    tgrid::TimeGrid{T}
end

getdimension(grid::NGrid) = getdimension(grid.kgrid)

function printparamsSI(g::NGrid, us::UnitScaling; digits = 3)
    str = printparamsSI(g.tgrid, us; digits = digits)
    str *= printparamsSI(g.kgrid, us; digits = digits)
    return str
end

function Base.show(io::IO, ::MIME"text/plain", g::NGrid)
    buf = IOBuffer()
    str = ""
    Base.show(buf, MIME"text/plain"(), g.kgrid)
    str *= String(take!(buf))
    Base.show(buf, MIME"text/plain"(), g.tgrid)
    str *= String(take!(buf))
	println(io, "NGrid:")
	print(io, str)
end