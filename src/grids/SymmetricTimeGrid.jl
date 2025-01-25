export SymmetricTimeGrid

"""
	SymmetricTimeGrid{T}(dt, t0)

Time discretzation of a [`Simulation`](@ref) spanning (-t0,t0) in steps of dt.

# Fields
- `dt::T`: timestep in internal dimensionless units (see [`UnitScaling`](@ref)).
- `t0::T`: .

# See also
[`Simulation`](@ref), [`SymmetricTimeGrid`](@ref), [`CartesianKGrid1d`](@ref),
[`CartesianKGrid2d`](@ref)
"""
struct SymmetricTimeGrid{T <: Real} <: TimeGrid{T}
    dt::T
    t0::T
end
SymmetricTimeGrid(dt::Real,t0::Real)    = SymmetricTimeGrid(promote(dt,t0)...)

gettsamples(tgrid::SymmetricTimeGrid)   = -abs(tgrid.t0):tgrid.dt:abs(tgrid.t0)
getnt(tgrid::SymmetricTimeGrid)         = length(gettsamples(tgrid))
gettspan(tgrid::SymmetricTimeGrid)      = (gettsamples(tgrid)[1],gettsamples(tgrid)[end])
getdt(tgrid::SymmetricTimeGrid)         = tgrid.dt

for func ∈ (:gettsamples,:getnt,:gettspan,:getdt)
    @eval(Damysos,$func(s::Simulation) = $func(s.grid))
    @eval(Damysos,$func(g::NGrid)       = $func(g.tgrid))
end

function printparamsSI(tgrid::SymmetricTimeGrid, us::UnitScaling; digits = 3)
    params  = [("dt",tgrid.dt), ("t0",tgrid.t0)]
    strings = [param_string(p[1],wavenumberSI(p[2], us),p[2]; digits = digits) for p in params]
    return strings[1] * "\n" * strings[2] *"\n" 
end

function Base.show(io::IO, ::MIME"text/plain", tgrid::SymmetricTimeGrid)
	println(io, "SymmetricTimeGrid:")
	print(io, printfields_generic(tgrid) |> prepend_spaces)
end