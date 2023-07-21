
function ensurepath(path::String)
    try
        if !isdir(path)
            mkpath(path)
        end
    catch e
        @warn "could not create $path" e
        return false
    end
    return true
end


function savedata(sim::Simulation{T}) where {T<:Real}

    dat         = DataFrame(t=getparams(sim).tsamples)
    datapath    = sim.datapath

    if !ensurepath(dirname(joinpath(datapath,"data.csv")))
        @info "Using working dir instead"
        datapath = ""
    end
    

    for o in sim.observables
        addproperobs!(dat,o)
        saveimproperobs(o)
    end

    CSV.write(joinpath(datapath,"data.csv"),dat)
    @info "Saved Simulation data at "*joinpath(datapath,"data.csv")

    return nothing
end

function loaddata(sim::Simulation)
    return DataFrame(CSV.File(joinpath(sim.datapath,"data.csv")))
end

function addproperobs!(dat::DataFrame,v::Velocity)
    dat.vx          = v.vx
    dat.vxintra     = v.vxintra
    dat.vxinter     = v.vxinter
    # skip for 1d
    if length(v.vy) == length(v.vx)
        dat.vy          = v.vy
        dat.vyintra     = v.vyintra
        dat.vyinter     = v.vyinter
    end
    
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

    if !ensurepath(dirname(filepath))
        @info "Using working dir instead"
        filepath = basename(filepath)
    end
    try
        touch(filepath)
        file = open(filepath,"w")
        write(file,"$object")
        close(file)
    catch e
        @warn "Could not save to $filepath ",e
    end
    
end


function load(filepath::String)

    file    = open(filepath,"r")
    code    = read(file,String)
    close(file)

    return eval(Meta.parse(code))
end

