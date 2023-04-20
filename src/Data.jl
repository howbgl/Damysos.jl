
function ensurepath(path::String)
    if !isdir(path)
        mkpath(path)
    end
end


function savedata(sim::Simulation{T}) where {T<:Real}

    dat         = DataFrame(t=getparams(sim).tsamples)
    
    ensurepath(dirname(sim.datapath))

    for o in sim.observables
        addproperobs!(dat,o)
        saveimproperobs(o)
    end

    CSV.write(sim.datapath*"data.csv",dat)
    @info "Saved Simulation data at "*sim.datapath*"data.csv"

    return nothing
end

function loaddata(sim::Simulation)
    return DataFrame(CSV.File(sim.datapath*"data.csv"))
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

    filename = "simulation.meta"
    save(sim.datapath*filename,sim)
    @info "Simulation metadata saved at "*sim.datapath*filename
end


function savemetadata(ens::Ensemble)
    
    filename =  "ensemble.meta"
    save(ens.datapath*filename,ens)
    @info "Ensemble metadata save at "*ens.datapath*filename
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

