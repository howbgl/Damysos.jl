@inline function vector_of_svec_to_matrix(u::Vector{SVector{N,T}}) where {N,T}
    return reshape(reinterpret(T,u),(N,:))
end

export subdivide_vector
function subdivide_vector(vec::AbstractVector,step::Integer,overlap::Integer=0)

    step <= 0 && throw(ArgumentError("step must be positive integer"))
    overlap <= -step && throw(ArgumentError("overlap must be larger than -step"))

    return (vec[1+d:minimum((end,step+d+overlap))] for d in 0:step:length(vec))
end

function makeindices_kspace_linearsubdiv(
    kxsamples::AbstractVector{<:Number},
    kysamples::AbstractVector{<:Number},
    kchunksize::Integer=4096)

    fullrange = 1:length(CartesianIndices((length(kxsamples),length(kysamples))))
    return subdivide_vector(fullrange,kchunksize)
end

function make_kspace_tiling(
    kxsamples::AbstractVector{<:Number},
    kysamples::AbstractVector{<:Number},
    kchunksize::Integer=4096;
    kxoverlap=0,
    kyoverlap=0)
    
    nk              = largest_root_below(kchunksize)
    kchunksize      = nk^2
    return make_kspace_tiling(kxsamples,kysamples,nk,nk;
        kxoverlap=kxoverlap,kyoverlap=kyoverlap)
end

function make_kspace_tiling(
    kxsamples::AbstractVector{<:Number},
    kysamples::AbstractVector{<:Number},
    nkxtile::Integer,
    nkytile::Integer;
    kxoverlap=0,
    kyoverlap=0)

    kxtiles = subdivide_vector(kxsamples,nkxtile,kxoverlap)
    kytiles = subdivide_vector(kysamples,nkytile,kyoverlap)

    kxtiling = [kxs for kxs in kxtiles for _ in kytiles]
    kytiling = [kys for _ in kxtiles for kys in kytiles]
    return (kxtiling,kytiling)
end


function padvecto_overlap(kybatches::AbstractVector{V}) where {V<:AbstractVector}
    padvecto_overlap!(deepcopy(kybatches))
    return kybatches
end

function padvecto_overlap!(kybatches::Vector{V}) where {V<:AbstractVector}
    for i in 2:length(kybatches)
        push!(kybatches[i-1], kybatches[i][1])
    end
    return kybatches
end

function nestedcount(x::Vector)
    if isprimitivetype(eltype(x))
        return length(x)
    else
        return sum((nestedcount(el) for el in x))
    end
end

largest_root_below(n::Integer) = floor(Int, sqrt(n))

export find_files_with_name
function find_files_with_name(root_dir::String, target_name::String)
    file_paths = Vector{String}()
    
    for entry in readdir(root_dir)
        full_path = joinpath(root_dir, entry)
        if isfile(full_path) && occursin(target_name, entry)
            push!(file_paths, full_path)
        elseif isdir(full_path)
            subfiles = find_files_with_name(full_path, target_name)
            append!(file_paths, subfiles)
        end
    end
    
    return file_paths
end

function stringexpand_vector(v::AbstractVector)
    str = ""
    for i in eachindex(v)
        if i == length(v) # drop last underscore
            str *= "$(v[i])"
        else
            str *= "$(v[i])_"
        end
    end
    return str
end

function stringexpand_nt(nt::NamedTuple)
    str = ""
    for (k, v) in pairs(nt)
        str *= "$k: $v\n"
    end
    return str
end

function prepend_spaces(str::AbstractString,n_spaces::Int64=4)
    lines = split(str, '\n')
    indented_lines = [repeat(" ",n_spaces)*"$line" for line in lines]
    indented_str = join(indented_lines, '\n')
    return indented_str
end


droplast(path::AbstractString) = joinpath(splitpath(path)[1:end-1]...)

function chopto_length_from_front(s::AbstractString,l::Integer)
    if length(s) > l
        return s[end-l:end]
    else
        return s
    end
end

export random_word
function random_word()::String
    lines = readlines("words.txt")
    
    # Check if the file is empty
    if isempty(lines)
        return ""
    end
    
    random_index = rand(1:length(lines))
    # Remove spaces from the selected line using a regular expression
    selected_line = replace(lines[random_index], r"\s+" => "")
    
    return selected_line
end


function try_execute_n_times(f::Function, n::Int, arg; wait_time::Real=10.0)

    success = false
    for i in 1:n
        try
            f(arg)
            success = true
            break  # Break out of the loop if successful attempt
        catch e
            @warn "Error caught on attempt $i: $e"
        end

        if i < n && wait_time > 0
            sleep(wait_time)
        end
    end
    return success
end

export ensurepath
function ensurepath(paths::Vector{String}; n_tries::Int=3, wait_time::Real=10.0)

    for path in paths
        success = ensurepath(path; n_tries=n_tries, wait_time=wait_time)
        if success
            return (true, path)
        end
    end

    @warn "None of the given paths could be created."
    return (false, "")
end

function ensurepath(path::String; n_tries::Int=3, wait_time::Real=10.0)

    @debug "Attempting to create \"...$path\""
    @debug "Full path: $path"
    success = false
    if !isdir(path)
        success = try_execute_n_times(mkpath, n_tries, path; wait_time=wait_time)
    else
        @debug "\"$path\" already exists."
        return true
    end

    if success
        @debug "\"$path\" created."
        return true
    else
        @warn "Could not create \"$path\""
        return false
    end
end



function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol,
    range::AbstractVector{T}; id="") where {T<:Real}

    return parametersweep(sim,comp,[param],[(r,) for r in range];id=id)
end

function parametersweep(
    sim::Simulation{T},
    comp::SimulationComponent{T},
    params::Vector{Symbol},
    range::Vector{Tuple{Vararg{T, N}}};
    id="",
    plotpath="",
    datapath="") where {T<:Real,N}

    random_name  = "$(today())_" * random_word()
    plotpath     = plotpath == "" ? droplast(sim.plotpath) : plotpath
    datapath     = datapath == "" ? droplast(sim.datapath) : datapath
    ensname      = "Ensemble[$(length(range))]($(sim.dimensions)d)" 
    ensname      *= getshortname(sim.hamiltonian) *"_"* getshortname(sim.drivingfield) * "_"
    ensname      *= stringexpand_vector(params)*"_sweep_" * random_name
    id           = id == "" ? stringexpand_vector(params)*"_sweep_" * random_name : id
    

    sweeplist = Vector{Simulation{T}}(undef, length(range))
    for i in eachindex(sweeplist)

        name = ""
        for (p, v) in zip(params, range[i])
            name *= "$p=$(v)_"
        end
        name = name[1:end-1] # drop last underscore

        new_h = deepcopy(sim.hamiltonian)
        new_df = deepcopy(sim.drivingfield)
        new_p = deepcopy(sim.numericalparams)

        if comp isa Hamiltonian{T}
            for (p, v) in zip(params, range[i])
                new_h = set(new_h, PropertyLens(p), v)
            end
        elseif comp isa DrivingField{T}
            for (p, v) in zip(params, range[i])
                new_df = set(new_df, PropertyLens(p), v)
            end
        elseif comp isa NumericalParameters{T}
            for (p, v) in zip(params, range[i])
                new_p = set(new_p, PropertyLens(p), v)
            end
        end
        sweeplist[i] = Simulation(new_h, new_df, new_p, deepcopy(sim.observables),
            sim.unitscaling, sim.dimensions, name,
            joinpath(datapath, ensname, name * "/"),
            joinpath(plotpath, ensname, name * "/"))
    end


    return Ensemble(
        sweeplist,
        id,
        joinpath(datapath, ensname * "/"),
        joinpath(plotpath, ensname * "/"))
end


function resize_obs!(sim::Simulation{T}) where {T<:Real}

    sim.observables .= [resize(o, sim.numericalparams) for o in sim.observables]
end

function maximum_k(df::DrivingField)
    @warn "using fallback for maximum k value of DrivingField!"
    return df.eE / df.ω
end
maximum_k(df::GaussianAPulse) = df.eE / df.ω

function semiclassical_interband_range(h::GappedDirac, df::DrivingField)
    ϵ = getϵ(h)
    ωmin = 2.0 * ϵ(0.0, 0.0)
    kmax = maximum_k(df)
    ωmax = 2.0 * ϵ(kmax, 0.0)
    min_harm = ωmin / df.ω
    max_harm = ωmax / df.ω
    println("Approximate range of semiclassical interband: ", min_harm, " to ",
        max_harm, " (harmonic number)")
end
