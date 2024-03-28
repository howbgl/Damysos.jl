
const MIN_CHUNK_SIZE = 128

function pairwise_reduction(
    sols::Vector,
    ts::Vector{<:Real},
    ks::Vector{<:SVector{2,<:Real}},
    bzmask::Function,
    obsfunction::Function;
    min_chunk_size=MIN_CHUNK_SIZE)
    
    length(sols) != length(ks) && throw(ArgumentError(
        "solution vector and k-point vector have unequal lengths"))

    n = length(sols)
    if n <= min_chunk_size
        o = [bzmask.(k[1],k[2],ts) .* obsfunction.(s,k[1],k[2],ts) for (s,k) in (sols,ks)]
    end
end

function pairwise_sum(x::Union{Vector,EnsembleSolution},min_chunk_size=MIN_CHUNK_SIZE)
    
    n = length(x)
    if n <= min_chunk_size
        return sum(x)
    else
        m = floor(Int64,n/2)
        s1 = Dagger.@spawn pairwise_sum(x[1:m])
        s2 = Dagger.@spawn pairwise_sum(x[m+1:end])
        return fetch(s1) + fetch(s2)
    end
end

