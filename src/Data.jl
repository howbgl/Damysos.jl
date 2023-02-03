
function savedata(sim::Simulation{T},obs) where {T<:Real}

    dat         = DataFrame()
    names       = getnames_obs(sim)
    arekres     = arekresolved(sim)
    filename    = getfilename(sim)
    if !isdir(sim.datapath*filename)
        mkpath(sim.datapath*filename)
    end

    if length(eachindex(names)) == length(eachindex(obs))   
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(names[i]),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end 
    else
        println("length(eachindex(names)) != length(eachindex(obs)) in savedata(sim::Simulation{T},obs)")
        println("Using numbers as names instead...")
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(i),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end
    end

    CSV.write(sim.datapath*filename*"/data.csv",dat)
    println("Saved Simulation data at ",sim.datapath*filename,"/data.csv")

    return nothing
end
