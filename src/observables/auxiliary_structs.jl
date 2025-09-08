export DensityMatrix


"""    DensityMatrix{T<:Real,U}

Holds the density matrix elements on a k grid.

# Fields
- `kgrid::KGrid{T}`: k grid on which the density matrix is defined
- `density_matrix::Vector{U}`: density matrices at each k-point

# See also
[`DensityMatrixSnapshots`](@ref DensityMatrixSnapshots)
"""
struct DensityMatrix{T<:Real,U}
    kgrid::KGrid{T}
    density_matrix::Vector{U}
end

DensityMatrix(sim::Simulation)         = DensityMatrix(sim.liouvillian,sim.grid)
DensityMatrix(l::Liouvillian,g::NGrid) = DensityMatrix(l,g.kgrid)

function DensityMatrix(l::TwoBandDephasingLiouvillian,kgrid::KGrid{T}) where {T<:Real}
    nk = getnk(kgrid)
    dm = [@SMatrix zeros(Complex{T}, 2, 2) for _ in 1:nk]
    return DensityMatrix(kgrid, dm)
end

function DensityMatrix(::SimulationComponent{T}) where {T<:Real}
    return DensityMatrix(KGridEmpty{T}(), empty(zeros(T,1)))    
end

function DensityMatrix(kgrid::KGrid, dmatrices::Array{<:Number,3})
    n,m = size(dmatrices,1), size(dmatrices,2)
    return DensityMatrix(kgrid, [SMatrix{n,m}(dmatrices[:,:,i]) for i in 1:size(dmatrices,3)])
end

resize(dm::DensityMatrix,sim::Simulation) =  DensityMatrix(sim.liouvillian,sim.grid)   
normalize!(dm::DensityMatrix,norm::Real)  = nothing # is always normalized per mode

function +(dm1::DensityMatrix{T,U},dm2::DensityMatrix{V,W}) where {T,U,V,W}
    @argcheck dm1.kgrid == dm2.kgrid "k-grids must be the same"
    return DensityMatrix(dm1.kgrid, dm1.density_matrix .+ dm2.density_matrix)
end
function -(dm1::DensityMatrix{T,U},dm2::DensityMatrix{V,W}) where {T,U,V,W}
    @argcheck dm1.kgrid == dm2.kgrid "k-grids must be the same"
    return DensityMatrix(dm1.kgrid, dm1.density_matrix .- dm2.density_matrix)
end
function *(dm::DensityMatrix{T,U},α::Real) where {T,U}
    return DensityMatrix(dm.kgrid, dm.density_matrix .* α)
end
*(α::Real,dm::DensityMatrix) = dm * α

function zero(dm::DensityMatrix{T,U}) where {T,U}
    return DensityMatrix(dm.kgrid, [zero(d) for d in dm.density_matrix])
end

function Base.isapprox(
    dm1::DensityMatrix{T,U},
    dm2::DensityMatrix{V,W};
    atol = 0,
    rtol = atol > 0 ? 0 : √eps(promote_type(eltype(T),eltype(V))),
    nans::Bool=false) where {T,U,V,W}

    @argcheck dm1.kgrid == dm2.kgrid "k-grids must be the same"

    return all(isapprox.(dm1.density_matrix, dm2.density_matrix; atol=atol, rtol=rtol, nans=nans))    
end

function count_nans(dm::DensityMatrix)
    return sum(sum(isnan.(mat)) for mat in dm.density_matrix)
end

function toarray(dm::DensityMatrix{T,U}) where {T,U<:SMatrix}
    return vector_of_smat_to_array(dm.density_matrix)
end

function Base.show(io::IO, ::MIME"text/plain", dm::DensityMatrix{T,U}) where {T,U}

    buf         = IOBuffer()
    nkpoints    = length(getksamples(dm.kgrid))

	println(io, "$nkpoints-element DensityMatrix{$T,$U}:")
    ds       = get(io, :displaysize, displaysize(io))
    io = IOContext(io, 
        :displaysize => (ds[1]-1,ds[2]), 
        :typeinfo => eltype(dm.density_matrix),
        :compact => true)
    io_recur = IOContext(io, :SHOWN_SET => dm.density_matrix)
    Base.print_array(io_recur, dm.density_matrix)
end
