
export ensuredirpath
export ensurefilepath
export find_files_with_name
export random_word
export replace_expression!

@inline cartesianindex2dx(i,n) = 1 + ((i-1) % n)
@inline cartesianindex2dy(i,n) = 1 + ((i-1) ÷ n)
@inline caresianindex2d(i,n)   = (cartesianindex2dx(i,n),cartesianindex2dy(i,n))

@inline function vector_of_svec_to_matrix(u::Vector{SVector{N,T}}) where {N,T}
    return reshape(reinterpret(T,u),(N,:))
end


@inline function get_cartesianindices_kgrid(
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    return CartesianIndices((length(kxsamples),length(kysamples)))
end

@inline function getkgrid_index(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    return get_cartesianindices_kgrid(kxsamples,kysamples)[i]
end

@inline function getkgrid_index(i::Integer,nkx::Integer,nky::Integer)
    return caresianindex2d(i,nkx)
end

@inline function getkgrid_point(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = caresianindex2d(i,length(kxsamples))

    return SA[kxsamples[idx[1]],kysamples[idx[2]]]
end


@inline function getkgrid_point_kx(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = getkgrid_index(i,kxsamples,kysamples)

    return kxsamples[idx[1]]
end
@inline function getkgrid_point_ky(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = getkgrid_index(i,kxsamples,kysamples)

    return kysamples[idx[2]]
end


function replace_expression!(e::Expr, old::Union{Expr,Symbol}, new::Union{Expr,Symbol})
    for (i,a) in enumerate(e.args)
        if a==old
            e.args[i] = new
        elseif a isa Expr
            replace_expression!(a, old, new)
        end
        ## otherwise do nothing
    end
    return e
end

function replace_expressions!(e::Expr,rules::Dict)
    for (old,new) in rules
        replace_expression!(e,old,new)
    end
    return e
end

function subdivide_vector(vec::AbstractVector,basesize::Integer)

    batches = Vector{Vector{eltype(vec)}}(undef, 0)
    buffer  = Vector{eltype(vec)}(undef,0)

    for (i,el) in enumerate(vec)
        push!(buffer,el)
        if length(buffer)==basesize || i==length(vec)
            push!(batches,deepcopy(buffer))
            buffer  = Vector{eltype(vec)}(undef,0)
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

function apply_with_longest!(x::Vector{<:Vector{<:Number}}, f!::Function)
    # Find the longest vector in x
    longest_vec = x[argmax(length.(x))]
    
    # Apply f! to each vector in x with the longest vector
    for vec in x
        f!(vec, longest_vec)
    end
end

function upsample!(x::Vector{<:Vector{<:Number}})
    return apply_with_longest!(x, upsample!)
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

function downsample!(x::Vector{Vector{<:Number}})
    return apply_with_longest!(x,downsample!)
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
    str = []
    for (k, v) in pairs(nt)
        push!(str,"$k: $v")
    end
    return join(str,'\n')
end

function stringexpand_2nt(nt1::NamedTuple,nt2::NamedTuple)
    str = ""
    for key in intersect(keys(nt1),keys(nt2))
        v1  = getfield(nt1,key)
        v2  = getfield(nt2,key)
        str *= "$key: $v1 ($v2)\n"
    end
    return str
end

function escape_underscores(input::AbstractString)
    output = replace(input, r"_"=>"\\_")
    return output
end


function prepend_spaces(str::AbstractString,n_spaces::Int64=1)
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

function appendtoname(filepath::String,s::String)
    path,ext = splitext(filepath)
    return path*s*ext
end


function rename_file_if_exists(filepath::String)
    if isfile(filepath)
        newpath = appendtoname(filepath,"_old_"*basename(tempname()))
        mv(filepath,newpath)
        @warn "Renamed $filepath to $newpath"
    end
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

function ensurefilepath(args...; n_tries::Int=3, wait_time::Real=10.0)
    return ensurepath(isfile,p -> mkpath(dirname(p)),args...;n_tries=n_tries,wait_time=wait_time)
end

function ensuredirpath(args...; n_tries::Int=3, wait_time::Real=10.0)
    return ensurepath(isdir,mkpath,args...;n_tries=n_tries,wait_time=wait_time)
end


function ensurepath(check::Function, make::Function, paths::Vector{String};
    n_tries::Int=3,
    wait_time::Real=10.0)

    for path in paths
        success = ensurepath(check, make, path; n_tries=n_tries, wait_time=wait_time)
        if success
            return (true, path)
        end
    end

    @warn "None of the given paths could be created."
    return (false, "")   
end

function ensurepath(check::Function, make::Function, path::String; 
    n_tries::Int=3, 
    wait_time::Real=10.0)

    @debug "Attempting to create \"...$path\""
    @debug "Full path: $path"
    success = false
    if !check(path)
        success = try_execute_n_times(make, n_tries, path; wait_time=wait_time)
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

"Fits a straight line through a set of points and returns an anonymous fit-function"
function linear_fit(x, y)

    
    sx = sum(x)
    sy = sum(y)

    m = length(x)

    sx2 = zero(sx.*sx)
    sy2 = zero(sy.*sy)
    sxy = zero(sx*sy)

    for i = 1:m
        sx2 += x[i]*x[i]
        sy2 += y[i]*y[i]
        sxy += x[i]*y[i]
    end

    a0 = (sx2*sy - sxy*sx) / ( m*sx2 - sx*sx )
    a1 = (m*sxy - sx*sy) / (m*sx2 - sx*sx)

    # return (a0, a1)
    return x -> a0 + a1*x
end

