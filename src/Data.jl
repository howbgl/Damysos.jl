
function ensurepath(path::String)
    if !isdir(path)
        mkpath(path)
    end
end


function savedata(sim::Simulation{T}) where {T<:Real}

    dat         = DataFrame(t=getparams(sim).tsamples)
    filename    = getname(sim)*"/data.csv"
    
    ensurepath(dirname(sim.datapath*filename))

    for o in sim.observables
        addproperobs!(dat,o)
        saveimproperobs(o)
    end

    CSV.write(sim.datapath*filename,dat)
    println("Saved Simulation data at ",sim.datapath*filename)

    return nothing
end

function addproperobs!(dat::DataFrame,v::Velocity)
    dat.vx          = v.vx
    dat.vxintra     = v.vxintra
    dat.vxinter     = v.vxinter
end

function addproperobs!(dat::DataFrame,occ::Occupation)
    dat.cbocc   = occ.cbocc
end

function saveimproperobs(v::Velocity)
    return nothing
end


function saveimproperobs(occ::Occupation)
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

