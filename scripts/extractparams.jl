using Damysos,DataFrames

function extract_values(input::AbstractString)
    matches = Dict{String, Float64}()
    pattern = r"(\w+)\s*(?:=|:)\s*([\d\.eE+-]+)(?:\s*.*?\(([\d\.eE+-]+)\))?"
    for match in eachmatch(pattern, input)
        name = match.captures[1]
        value = match.captures[end] == nothing ? parse(Float64, match.captures[2]) : parse(Float64, match.captures[3])
        matches[name] = value
    end
    return matches
end

function process(files::AbstractVector)
    dicts = []
    for f in files
        d = process(f)
        !isnothing(d) && push!(dicts,d)
    end
    return DataFrame(dicts)
end

function process(path::AbstractString)

    filecontent = read(path,String)
    if length(filecontent)<500 # skip old files
        @info "Skip old file"
        return
    end
    head,_,p    = split(filecontent,r"(?:First simulation|Last simulation)")

    headvals = extract_values(head)
    pvals    = extract_values(p)
    @show filecontent

    if headvals["rtol"]<1e-4
        return Dict(
            "rtol"=>headvals["rtol"],
            "atol"=>headvals["atol"],
            "ζ"=>pvals["ζ"],
            "γ"=>pvals["γ"],
            "dt"=>pvals["dt"],
            "dkx"=>pvals["dkx"])
    else
        @warn "rtol > 1e-4"
        return
    end

end

files = find_files_with_name(
    "/home/how09898/phd/data/hhgjl/inter-intra-cancellation",
    "combined-testresult.txt")

