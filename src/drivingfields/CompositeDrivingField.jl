
export CompositeDrivingField

struct CompositeDrivingField{N,T<:Real} <: DrivingField{T}
    fields::SVector{N,DrivingField{T}}
    prefactors::SVector{N,T}
end

function CompositeDrivingField(fields::SVector{N,DrivingField{T}}) where {T<:Real,N}
    return CompositeDrivingField(fields, @SVector ones(T, N))    
end

function CompositeDrivingField(
    fields::AbstractVector{<:DrivingField{T}},
    prefactors::AbstractVector{T} = ones(T, length(fields))) where {T<:Real}
    return CompositeDrivingField(SA{DrivingField{T}}[fields...],SA{T}[prefactors...])
end

function Base.:+(df1::DrivingField{T}, df2::DrivingField{U}) where {T,U}
    @argcheck T == U "Base.:+(df1::DrivingField{T}, df2::DrivingField{U})"
    return CompositeDrivingField(SA{DrivingField{T}}[df1, df2])
end

function Base.:+(df1::CompositeDrivingField{N,T}, df2::DrivingField{U}) where {N,T,U}
    @argcheck T == U "Base.:+(df1::CompositeDrivingField{N,T}, df2::DrivingField{U})"
    return CompositeDrivingField{N+1, T}(
        SA{DrivingField{T}}[df1.fields..., df2],
        SA{T}[df1.prefactors..., one(T)])    
end

function Base.:+(df1::DrivingField, df2::CompositeDrivingField)
    return df2 + df1    
end

function Base.:+(df1::CompositeDrivingField{N,T}, df2::CompositeDrivingField{M,U}) where {N,T,M,U}
    @argcheck T == U "Base.:+(df1::CompositeDrivingField{N,T}, df2::CompositeDrivingField{M,U})"
    return CompositeDrivingField{N+M, T}(
        SA{DrivingField{T}}[df1.fields..., df2.fields...],
        SA{T}[df1.prefactors..., df2.prefactors...])    
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
