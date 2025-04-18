using Damysos,DataFrames

function extract_values(input::AbstractString)
    matches = Dict{String, Float64}()
    pattern = r"(\w+)\s*(?:=|:)\s*([\d\.eE+-]+)(?:\s*.*?\(([\d\.eE+-]+)\))?"
    for match in eachmatch(pattern, input)
        name = match.captures[1]
        value = isnothing(match.captures[end]) ? parse(Float64, match.captures[2]) : parse(Float64, match.captures[3])
        matches[name] = value
    end
    return matches
end

function process(files::AbstractVector,args...)
    dicts = []
    for f in files
        d = process(f,args...)
        !isnothing(d) && push!(dicts,d)
    end
    return DataFrame(dicts)
end

function process(path::AbstractString,rtolthresh=1e-4)

    filecontent = read(path,String)
    if length(filecontent)<500 # skip old files
        @info "Skip old file"
        return
    end
    head,_,p    = split(filecontent,r"(?:First simulation|Last simulation)")

    headvals = extract_values(head)
    pvals    = extract_values(p)

    if headvals["rtol"]<rtolthresh
        return Dict(
            "rtol"=>headvals["rtol"],
            "atol"=>headvals["atol"],
            "ζ"=>pvals["ζ"],
            "γ"=>pvals["γ"],
            "dt"=>pvals["dt"],
            "dkx"=>pvals["dkx"],
            "M"=>pvals["M"],
            "kxmax"=>pvals["kxmax"],
            "nkx"=>pvals["nkx"],
            "eE"=>pvals["eE"],
            "ω"=>pvals["ω"],
            "amax"=>pvals["eE"] ./ pvals["ω"])
    else
        @warn "rtol > $rtolthresh"
        return
    end

end

files = find_files_with_name(
    "/home/how09898/phd/data/hhgjl/inter-intra-cancellation",
    "combined-testresult.txt")

