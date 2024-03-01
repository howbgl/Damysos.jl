export load
export loaddata
export save
export savedata
export savemetadata


function savedata(sim::Simulation)

    @info "Saving simulation data"
    @debug "datapath = \"$(sim.datapath)\""

    tsamples    = getparams(sim).tsamples
    dat         = DataFrame(t=tsamples)
    
    for o in sim.observables
        addproperobs!(dat,o)
        add_drivingfield!(dat,sim.drivingfield,tsamples)
        saveimproperobs(o)
    end
    
    datapath            = sim.datapath
    altpath             = joinpath(pwd(),basename(datapath))
    (success,datapath)  = ensurepath([datapath,altpath])

    if success
        CSV.write(joinpath(datapath,"data.csv"),dat)
        @debug "Saved Simulation data at\n\"$datapath\""
    else
        @warn "Could not save data.csv."
    end

    return nothing
end

function loaddata(sim::Simulation)
    return DataFrame(CSV.File(joinpath(sim.datapath,"data.csv")))
end

function savedata(
    result::ConvergenceTestResult,
    path::String,
    altpath=joinpath(pwd(),"testresult.txt"))
    
    filename       = splitdir(path)[end]
    (success,path) = ensurepath([path,altpath])
    if success
        save(joinpath(path,filename),repr("text/plain",result))
        @info "Saved convergence test result at $path"
    end
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

function add_drivingfield!(dat::DataFrame,df::DrivingField,tsamples::AbstractVector{<:Real})
    
    fx = get_efieldx(df)
    fy = get_efieldy(df)
    ax = get_vecpotx(df)
    ay = get_vecpoty(df)

    dat.fx = fx.(tsamples)
    dat.fy = fy.(tsamples)
    dat.ax = ax.(tsamples)
    dat.ay = ay.(tsamples)
end

function savemetadata(sim::Simulation)

    filename            = "simulation.meta"
    altpath             = joinpath(pwd(),basename(sim.datapath))
    (success,datapath)  = ensurepath([sim.datapath,altpath])
    if success
        if save(joinpath(datapath,filename),sim)
            @debug "Simulation metadata saved at \""*joinpath(datapath,filename)*"\""
            return
        end
    end
    
    @warn "Could not save simulation metadata."
end


function savemetadata(ens::Ensemble)
    
    filename            =  "ensemble.meta"
    altpath             = joinpath(pwd(),basename(ens.datapath))
    (success,datapath)  = ensurepath([ens.datapath,altpath])
    if success
        if save(joinpath(datapath,filename),ens)
            @debug "Ensemble metadata saved at \""*joinpath(datapath,filename)*"\""
            return 
        end
    end
    
    @warn "Could not save ensemble metadata."
end


function save(filepath::String,object)
    
    try
        touch(filepath)
        file = open(filepath,"w")
        write(file,"$object")
        close(file)
    catch e
        @warn "Could not save to $filepath ",e
        return false
    end
    return true
end


function load(filepath::String)

    file    = open(filepath,"r")
    code    = read(file,String)
    close(file)

    return eval(Meta.parse(code))
end

