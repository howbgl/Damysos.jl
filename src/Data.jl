
function ensurepath(path::String)
    if !isdir(path)
        mkpath(path)
    end
end


function savedata(sim::Simulation{T},obs) where {T<:Real}

    dat         = DataFrame()
    names       = getnames_obs(sim)
    arekres     = arekresolved(sim)
    filename    = getname(sim)*"/data.csv"

    @show sim.datapath
    
    ensurepath(dirname(sim.datapath*filename))

    if length(eachindex(names)) == length(eachindex(obs))   
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(names[i]),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end 
    else # Fallback option if names don't match data: use numbers as names
        println("length(eachindex(names)) != length(eachindex(obs))\
                 in savedata(sim::Simulation{T},obs)")
        println("Using numbers as names instead...")
        for i in eachindex(names)
            if arekres[i] == false
                setproperty!(dat,Symbol(i),obs[i])
            else
                println("Skip saving k-resolved observables for now...")
            end
        end
    end

    CSV.write(sim.datapath*filename,dat)
    println("Saved Simulation data at ",sim.datapath*filename)

    return nothing
end


function savemetadata(sim::Simulation)

    filename = getname(sim) * "/simulation.meta"
    save(sim.datapath*filename,sim)
    println("Simulation metadata saved at ",sim.datapath*filename)
end


function savemetadata(ens::Ensemble)
    
    filename = getname(ens) * "/ensemble.meta"
    save(ens.datapath*filename,ens)
    println("Ensemble metadata save at ",ens.datapath*filename)
end


function save(filepath::String,object)
    ensurepath(dirname(filepath))
    touch(filepath)
    file = open(filepath,"w")
    write(file,"$object")
    close(file)
end


function load(filepath::String)

    file    = open(filepath,"r")
    code    = read(file,String)
    close(file)

    return eval(Meta.parse(code))
end

