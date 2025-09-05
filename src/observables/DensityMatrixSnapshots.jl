export DensityMatrixSnapshots
export DensityMatrix


"""    DensityMatrix{N<:Integer,T<:Real,U<:Number}

Holds the density matrix elements on a k grid.

# Fields
- `kgrid::KGrid{T}`: k grid on which the density matrix is defined
- `density_matrix::Vector{SVector{N,U}}`: density matrix elements for `N` bands 

# See also
[`DensityMatrixSnapshots`](@ref DensityMatrixSnapshots)
"""
struct DensityMatrix{N<:Integer,T<:Real,U<:Number}
    kgrid::KGrid{T}
    density_matrix::Vector{SVector{N,U}}
end

DensityMatrix(l::Liouvillian,g::NGrid) = DensityMatrix(l,g.kgrid)

function DensityMatrix(l::Liouvillian,kgrid::KGrid{T}) where {T<:Real}
    nk = getnk(kgrid)
    dm = [@SVector zeros(Complex{T}, getnbands(l)) for _ in 1:nk]
    @show typeof(dm)
    @show dm 
    return DensityMatrix(kgrid, dm)
end

function DensityMatrix(::SimulationComponent{T}) where {T<:Real}
    return DensityMatrix{2,T,Complex{T}}(
        KGridEmpty{T}(),
        empty([@SVector zeros(Complex{T},2)]))    
end

resize(dm::DensityMatrix,sim::Simulation) =  DensityMatrix(sim.liouvillian,sim.grid)   


"""
    DensityMatrixSnapshots{N<:Integer,T<:Real,U<:Number} <: Observable{T}

Holds snapshots in time of the density matrix on the k grid.

# Fields
- `tsamples::Vector{T}`: time samples at which the density matrix is stored
- `density_matrices::Vector{DensityMatrix{N,T,U}}`: vector of density matrices

# See also
[`Occupation`](@ref Occupation), [`Velocity`](@ref Velocity), [`DensityMatrix`](@ref DensityMatrix)
"""
struct DensityMatrixSnapshots{N<:Integer,T<:Real,U<:Number} <: Observable{T}
    tsamples::Vector{T}
    density_matrices::Vector{DensityMatrix{N,T,U}}
end
function DensityMatrixSnapshots(l::Liouvillian,g::NGrid;
    tsamples = collect(gettsamples(g)))

    return DensityMatrixSnapshots(tsamples, [DensityMatrix(l,g) for _ in tsamples])
end
function DensityMatrixSnapshots(c::SimulationComponent{T};
    tsamples = Vector{T}(undef, 0)) where {T<:Real}
    
    return DensityMatrixSnapshots(tsamples, [DensityMatrix(c) for _ in tsamples])
end

function resize(dms::DensityMatrixSnapshots,sim::Simulation)
    return DensityMatrixSnapshots(sim.liouvillian,sim.grid;
        tsamples = isempty(dms.tsamples) ? collect(gettsamples(sim.grid)) : dms.tsamples)
end