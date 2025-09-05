export DensityMatrixSnapshots
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

"""
    DensityMatrixSnapshots{N<:Integer,T<:Real} <: Observable{T}

Holds snapshots in time of the density matrix on the k grid.

# Fields
- `tsamples::Vector{T}`: time samples at which the density matrix is stored
- `density_matrices::Vector{DensityMatrix{T,U}}`: vector of density matrices

# See also
[`Occupation`](@ref Occupation), [`Velocity`](@ref Velocity), [`DensityMatrix`](@ref DensityMatrix)
"""
struct DensityMatrixSnapshots{T<:Real,U} <: Observable{T}
    tsamples::Vector{T}
    density_matrices::Vector{DensityMatrix{T,U}}
end
function DensityMatrixSnapshots(sim::Simulation;
    tsamples = collect(gettsamples(sim.grid)))
    return DensityMatrixSnapshots(tsamples, [DensityMatrix(sim) for _ in tsamples])
    
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
    @argcheck all([t ∈ gettsamples(sim) for t in dms.tsamples])
    return DensityMatrixSnapshots(sim.liouvillian,sim.grid;
        tsamples = isempty(dms.tsamples) ? collect(gettsamples(sim.grid)) : dms.tsamples)
end

function Base.append!(dms1::DensityMatrixSnapshots,dms2::DensityMatrixSnapshots)
    append!(dms1.tsamples,dms2.tsamples)
    append!(dms1.density_matrices,dms2.density_matrices)
    return dms1    
end

getshortname(::DensityMatrixSnapshots) = "DensityMatrixSnapshots"

function normalize!(dms::DensityMatrixSnapshots,norm::Real)
    normalize!.(dms.density_matrices,norm)
    return nothing
end

function +(dms1::DensityMatrixSnapshots,dms2::DensityMatrixSnapshots)
    @argcheck dms1.tsamples == dms2.tsamples "Time samples must be the same"
    return DensityMatrixSnapshots(dms1.tsamples,dms1.density_matrices .+ dms2.density_matrices)    
end

function -(dms1::DensityMatrixSnapshots,dms2::DensityMatrixSnapshots)
    @argcheck dms1.tsamples == dms2.tsamples "Time samples must be the same"
    return DensityMatrixSnapshots(dms1tsamples,dms1.density_matrices .- dms2.density_matrices)    
end

function *(dms::DensityMatrixSnapshots,α::Real)
    return DensityMatrixSnapshots(dms.tsamples,dms.density_matrices .* α)
end
*(α::Real,dms::DensityMatrixSnapshots) = dms * α

function zero(dms::DensityMatrixSnapshots)
    return DensityMatrixSnapshots(dms.tsamples, [zero(dm) for dm in dms.density_matrices])
end

function Base.isapprox(
    dms1::DensityMatrixSnapshots{T,U},
    dms2::DensityMatrixSnapshots{V,W};
    atol = 0,
    rtol = atol > 0 ? 0 : √eps(promote_type(eltype(T),eltype(V))),
    nans::Bool=false) where {T,U,V,W}

    @argcheck dms1.tsamples == dms2.tsamples "Time samples must be the same"

    return all(isapprox.(dms1.density_matrices, dms2.density_matrices; atol=atol, rtol=rtol, nans=nans))    
end

function buildobservable_vec_of_expr_cc_cv(sim::Simulation{T}, ::DensityMatrixSnapshots) where {T<:Real}
    return [:(SMatrix{2,2,Complex{$T}}(complex(real(cc)), conj(cv), cv, complex(real(1-cc))))]
end

function sum_observables!(
    dms::DensityMatrixSnapshots,
    funcs,
    ks,
    cc::Matrix{<:Complex},
    cv::Matrix{<:Complex},
    ts::Vector{<:Real},
    weigths::Matrix{<:Real})

    total_ks = getksamples(dms.density_matrices[1].kgrid)
    k_inds   = find_indices_in(ks, total_ks)

    dm   = funcs[1]
    inds = find_indices_in(dms.tsamples, ts) # ts[inds] == dms.tsamples ∈ ts
    _cc  = @view cc[:, inds]
    _cv  = @view cv[:, inds]
    _ts  = @view ts[inds]
    dmdata = dm.(_cc,_cv,ks,_ts')
    for i in inds
        dm_dest = dms.density_matrices[i].density_matrix
        dm_dest[k_inds] .= dmdata[:,i]
    end 
    return dms
end

function find_indices_in(x::Vector{T}, y::Vector{T}) where {T}
    sx = Set(x)
    inds = Int[]
    for (i, yi) in pairs(y)
        if yi in sx
            push!(inds, i)
        end
    end
    return inds
end

function count_nans(dms::DensityMatrixSnapshots)
    return sum(count_nans.(dms.density_matrices))
end


function Base.show(io::IO, ::MIME"text/plain", dms::DensityMatrixSnapshots{T,U}) where {T,U}

    buf        = IOBuffer()
    nsnapshots = length(dms.tsamples)
    nkpoints   = length(getksamples(dms.density_matrices[1].kgrid))

	println(io, "DensityMatrixSnapshots{$T,$U}:")
    println(io, " $nsnapshots tsamples")
    print(io, " $nkpoints k points at each timestep")
end
