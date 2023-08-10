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
        if i==length(v) # drop last underscore
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

function prepend_spaces(str::AbstractString)
    lines = split(str, '\n')
    indented_lines = ["    $line" for line in lines]
    indented_str = join(indented_lines, '\n')
    return indented_str
end


droplast(path::AbstractString) = joinpath(splitpath(path)[1:end-1]...)


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

function ensurepath(paths::Vector{String};n_tries::Int=3,wait_time::Real=10.0)

    for path in paths
        success = ensurepath(path;n_tries=n_tries,wait_time=wait_time)
        if success
            return (true,path)
        end
    end

    @warn "None of the given paths could be created."
    return (false,"")
end

function ensurepath(path::String;n_tries::Int=3,wait_time::Real=10.0)

    @info "Attempting to create $path"
    success     = false
    if !isdir(path)
        success = try_execute_n_times(mkpath,n_tries,path;wait_time=wait_time)
    else
        @info "$path already exists. Proceeding..."
        return true
    end

    if success
        @info "$path created. Proceeding..."
        return true
    else
        @warn "Could not create $path"
        return false
    end
end



function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, 
                        range::AbstractVector{T};id="") where {T<:Real}

    return parametersweep(sim,comp,[param],[[r] for r in range];id=id)
end

function parametersweep(sim::Simulation{T},comp::SimulationComponent{T},
    params::Vector{Symbol},range::Vector{Vector{T}};
    id="",
    plotpath="",
    datapath="") where {T<:Real}

    hashstring   = sprintf1("%x",hash([sim,comp,params,range]))
    plotpath     = plotpath == "" ? droplast(sim.plotpath) : plotpath
    datapath     = datapath == "" ? droplast(sim.datapath) : datapath
    ensname      = "Ensemble[$(length(range))]($(sim.dimensions)d)" 
    ensname      *= getshortname(sim.hamiltonian) *"_"* getshortname(sim.drivingfield) * "_"
    ensname      *= stringexpand_vector(params)*"_sweep_" * hashstring
    id           = id == "" ? stringexpand_vector(params)*"_sweep_" * hashstring : id
    

    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    for i in eachindex(sweeplist)

        name = ""
        for (p,v) in zip(params,range[i])
            name *= "$p=$(v)_"
        end
        name = name[1:end-1] # drop last underscore

        new_h  = deepcopy(sim.hamiltonian)
        new_df = deepcopy(sim.drivingfield)
        new_p  = deepcopy(sim.numericalparams)

        if comp isa Hamiltonian{T}
            for (p,v) in zip(params,range[i])
                new_h  = set(new_h,PropertyLens(p),v)
            end
        elseif comp isa DrivingField{T}
            for (p,v) in zip(params,range[i])
                new_df = set(new_df,PropertyLens(p),v)
            end
        elseif comp isa NumericalParameters{T}
            for (p,v) in zip(params,range[i])
                new_p  = set(new_p,PropertyLens(p),v)
            end
        end
        sweeplist[i] = Simulation(new_h,new_df,new_p,deepcopy(sim.observables),
                sim.unitscaling,sim.dimensions,name,
                joinpath(datapath,ensname,name*"/"),
                joinpath(plotpath,ensname,name*"/"))
    end


    return Ensemble(
                sweeplist,
                id,
                joinpath(datapath,ensname*"/"),
                joinpath(plotpath,ensname*"/"))
end

function maximum_k(df::DrivingField)
    @warn "using fallback for maximum k value of DrivingField!"
    return df.eE/df.ω
end
maximum_k(df::GaussianPulse) = df.eE/df.ω

function semiclassical_interband_range(h::GappedDirac,df::DrivingField)
    ϵ        = getϵ(h)
    ωmin     = 2.0*ϵ(0.0,0.0)
    kmax     = maximum_k(df)
    ωmax     = 2.0*ϵ(kmax,0.0)
    min_harm = ωmin/df.ω
    max_harm = ωmax/df.ω
    println("Approximate range of semiclassical interband: ",min_harm," to ",
            max_harm," (harmonic number)")
end
