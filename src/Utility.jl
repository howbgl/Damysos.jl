
export diffparams
export ensurepath
export find_files_with_name
export parametersweep
export random_word

function subdivide_vector(vec::AbstractVector{T}, basesize::U) where {T<:Real,U}

    batches = Vector{Vector{T}}(undef, 0)
    buffer  = Vector{T}(undef,0)

    for (i,el) in enumerate(vec)
        push!(buffer,el)
        if length(buffer)==basesize || i==length(vec)
            push!(batches,deepcopy(buffer))
            buffer  = Vector{T}(undef,0)
        end
    end

    return batches
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


function adjust_density(samples::Vector{<:Number}, desired_samples::Int) 
    n = length(samples)

    # Create an interpolation object for the original samples
    itp = interpolate(samples, BSpline(Cubic))

    # Calculate the interpolation positions for the desired number of samples
    interpolation_positions = range(1, stop = n, length = desired_samples)

    # Interpolate the values at the new positions
    interpolated_values = itp[interpolation_positions]

    return interpolated_values
end

function upsample!(a::Vector{<:Number},b::Vector{<:Number})
    
    la = length(a)
    lb = length(b)
    if la==lb
        return
    elseif la<lb
        buf = adjust_density(a,lb)
        resize!(a,lb)
        a .= buf
        return
    else
        buf = adjust_density(b,la)
        resize!(b,la)
        b .= buf
        return
    end
end

function downsample!(a::Vector{<:Number},b::Vector{<:Number})

    la = length(a)
    lb = length(b)

    if la == lb
        return
    elseif la > lb
        buf = adjust_density(a,lb)
        resize!(a,lb)
        a .= buf
        return
    else
        buf = adjust_density(b,la)
        resize!(b,la)
        b .= buf
        return
    end
end


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

stringexpand_vector(v::AbstractVector) = join(String.(v),"_")

function stringexpand_nt(nt::NamedTuple)
    str = ""
    for (k, v) in pairs(nt)
        str *= "$k: $v\n"
    end
    return str
end

function escape_underscores(input::AbstractString)
    output = replace(input, r"_"=>"\\_")
    return output
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
    if !ispath(path)
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



function parametersweep(
    sim::Simulation{T},
    comp::SimulationComponent{T},
    param::Symbol,
    range::AbstractVector{T};
    id="",
    plotpath="",
    datapath="") where {T<:Real}

    return parametersweep(sim,comp,[param],[(r,) for r in range];
        id=id,
        plotpath=plotpath,
        datapath=datapath)
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

"""
    diffparams(c1::SimulationComponent,c2::SimulationComponent)

Returns the parameters in c1 and c2, which have different values
"""
function diffparams(c1::SimulationComponent,c2::SimulationComponent)
    res = Symbol[]
    if !(c1 isa typeof(c2))
        throw(ArgumentError("""
        cannot compare the two different types $(typeof(c1)) and $(typeof(c1))"""))
    end
    for name in fieldnames(typeof(c1))
        getproperty(c1,name) != getproperty(c2,name) && push!(res,name)
    end
    return res
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
