export DensityMatrixSnapshots

"""
    DensityMatrixSnapshots{T<:Real,U} <: Observable{T}

Holds snapshots in time of the density matrix on the k grid.

# Fields
- `tsamples::Vector{T}`: time samples at which the density matrix is stored
- `density_matrices::Vector{DensityMatrix{T,U}}`: vector of density matrices of type `U`

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

function contract(dms::DensityMatrixSnapshots,ts::Vector{<:Real})
    
    inds = Int[]
    for (i,t) in enumerate(dms.tsamples)
        if t in ts
            push!(inds,i)
        end        
    end
    return DensityMatrixSnapshots(dms.tsamples[inds], dms.density_matrices[inds])
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

    @argcheck dms1.tsamples ≈ dms2.tsamples "Time samples must be the same"

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

    total_ks    = getksamples(dms.density_matrices[1].kgrid)
    k_inds      = find_indices_in(ks, total_ks)
    dm          = funcs[1]

    for (dm_dest, t, i) in zip(dms.density_matrices, dms.tsamples, 1:length(dms.tsamples))
        t_index         = find_index_nearest(t, ts)
        dms.tsamples[i] = ts[t_index]
        _cc             = @view cc[:, t_index]
        _cv             = @view cv[:, t_index]

        dm_dest.density_matrix[k_inds] .= dm.(_cc,_cv,ks,t)
    end

    return dms
end

function sum_observables!(
    dms::DensityMatrixSnapshots,
    funcs::Vector,
    d_kchunk::CuArray{<:SVector{2,<:Real}},
    d_us::CuArray{<:SVector{2,<:Complex}},
    d_ts::CuArray{<:Real,2},
    d_weigths::CuArray{<:Real,2},
    buf::CuArray{<:Real,2})
    
    total_ks = getksamples(dms.density_matrices[1].kgrid)
    k_inds   = find_indices_in(collect(d_kchunk), total_ks)
    dm       = funcs[1]
    
    # Views on GPU arrays are not permitted, require scalar indexing
    # Hence, copy everything to CPU first (DensityMatrixSnapshots are not performant on GPU)
    ks      = Array(d_kchunk)
    us      = Array(d_us)
    ts      = Array(d_ts[:,1])

    for (dm_dest, t, i) in zip(dms.density_matrices, dms.tsamples, 1:length(dms.tsamples))
        t_index             = find_index_nearest(t, ts)
        dms.tsamples[i]     = ts[t_index]
        _us                 = @view us[t_index, :]

        dm_dest.density_matrix[k_inds] .= dm.(_us,ks,t)
    end

    return dms
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
