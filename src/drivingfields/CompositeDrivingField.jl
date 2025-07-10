
export CompositeDrivingField

struct CompositeDrivingField{N,T<:Real} <: DrivingField{T}
    fields::SVector{N,DrivingField{T}}
    prefactors::SVector{N,T}
    function CompositeDrivingField{N,T}(fields::AbstractVector{<:DrivingField{T}},
                                        prefactors::AbstractVector{T}) where {N,T<:Real}
        @argcheck length(fields) == length(prefactors) "CompositeDrivingField: length mismatch"
        @argcheck N == length(fields) "CompositeDrivingField: N does not match length of fields"
        @argcheck !any(isa.(fields, CompositeDrivingField)) "CompositeDrivingField: fields must not contain CompositeDrivingFields"
        return new{N,T}(SA{DrivingField{T}}[fields...], SA{T}[prefactors...])
    end
end

function CompositeDrivingField(
    fields::SVector{N,DrivingField{T}},
    prefactors::SVector{N,T}) where {N,T<:Real}
    flat_fields, flat_prefactors = flatten_drivingfield_list(fields, prefactors)
    return CompositeDrivingField{N,T}(flat_fields, flat_prefactors)
end

function flatten_drivingfield_list(
    fields::AbstractVector{<:DrivingField{T}},
    prefactors::AbstractVector{T}) where {T<:Real}
    # Flatten a list of driving fields, removing any CompositeDrivingFields
    flat_fields     = DrivingField{T}[]
    flat_prefactors = T[]
    if any(isa.(fields, CompositeDrivingField))
        for (field,prefactor) in zip(fields, prefactors)
            if isa(field, CompositeDrivingField)
                append!(flat_fields, field.fields)
                append!(flat_prefactors, field.prefactors .* prefactor)
            else
                push!(flat_fields, field)
                push!(flat_prefactors, prefactor)
            end 
        end
        return flatten_drivingfield_list(flat_fields, flat_prefactors)
    else
        return SA{DrivingField{T}}[fields...], SA{T}[prefactors...]
    end
end

function CompositeDrivingField(
    fields::AbstractVector{<:DrivingField{T}},
    prefactors::AbstractVector{T} = ones(T, length(fields))) where {T<:Real}
    flat_fields, flat_prefactors = flatten_drivingfield_list(fields, prefactors)
    return CompositeDrivingField(SA{DrivingField{T}}[flat_fields...],SA{T}[flat_prefactors...])
end

function Base.:+(df1::DrivingField{T}, df2::DrivingField{U}) where {T,U}
    @argcheck T == U "Base.:+(df1::DrivingField{T}, df2::DrivingField{U})"
    return CompositeDrivingField(SA{DrivingField{T}}[df1, df2])
end

function Base.:-(df1::DrivingField{T}, df2::DrivingField{U}) where {T,U}
    @argcheck T == U "Base.:-(df1::DrivingField{T}, df2::DrivingField{U})"
    return CompositeDrivingField(SA{DrivingField{T}}[df1, df2], SA{T}[1,-1])    
end

function Base.:*(a::T, df::DrivingField{U}) where {T,U}
    @argcheck T == U "Base.:*(a::T, df::DrivingField{U})"
    return CompositeDrivingField(SA{DrivingField{T}}[df], SA{T}[a])
end

function Base.:*(a::T, df::CompositeDrivingField{N,U}) where {T,N,U}
    @argcheck T == U "Base.:*(a::T, df::CompositeDrivingField{N,U})"
    return CompositeDrivingField{N,T}(df.fields, a .* SA{T}[df.prefactors...])
end

function Base.:*(df::DrivingField, a::Real) 
    return Base.:*(a, df)
end

for func in [:get_vecpotx, :get_vecpoty, :get_efieldx, :get_efieldy]
    @eval(
        Damysos,
        function $func(df::CompositeDrivingField)
            funcs = [$func(f) for f in df.fields]
            return t -> sum([a*f(t) for (a,f) in zip(df.prefactors, funcs)])
        end
    )
end

for func in [:vecpotx, :vecpoty, :efieldx, :efieldy]
    @eval(
        Damysos,
        function $func(df::CompositeDrivingField, t::Real)
            field_contributions = $func.(df.fields, t)
            return df.prefactors ⋅ field_contributions
        end
    )
    @eval(
        Damysos,
        function $func(df::CompositeDrivingField)
            funcs = $func.(df.fields)
            exprs = [:($a * $f) for (a,f) in zip(df.prefactors, funcs)]
            return Expr(:call, :+, exprs...)
        end
    )
end

for func in [:maximum_vecpotx, :maximum_vecpoty, :maximum_efieldx, :maximum_efieldy]
    @eval(
        Damysos,
        function $func(df::CompositeDrivingField)
            return sum([abs(a) * $func(f) for (a,f) in zip(df.prefactors,df.fields)])
        end
    )
end

function Base.show(io::IO, ::MIME"text/plain", c::CompositeDrivingField)
	buf = IOBuffer()
	print(io, "CompositeDrivingField:\n")

    for i in 1:length(c.fields)
        prefactor = round(c.prefactors[i], sigdigits=3)
        print(io, " $i. $prefactor×")
        Base.show(buf, MIME"text/plain"(), c.fields[i])
        str = String(take!(buf))
        print(io, prepend_spaces(str) * "\n")
        
    end
end

function printparamsSI(
    df::CompositeDrivingField{N,T},
    us::UnitScaling;
    digits=4) where {N,T<:Real}

    str = ""
    for (i, field) in enumerate(df.fields)
        pars = printparamsSI(field, us; digits=digits)
        str *= append_to_parnames(pars, "_$i")
    end
    return str
end

function append_to_parnames(input::String,str::String)
    pattern = r"(.+?)\s*=\s*([^()]+)\s*(?:\(([\d\.]+)\))?"
    ret = ""
    
    for line in split(input, '\n')
        m = match(pattern, line)
        if !isnothing(m)
            name = m.captures[1]
            ret *= replace(line, name => name * str) * "\n"
        else
            ret *= line * "\n"
        end
    end
    return ret
end
