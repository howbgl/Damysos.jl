
export Ensemble

export getshortname
export make_ensemble_from_path

"""
    Ensemble{T}(simlist, id, datapath, plotpath)

A struct representing an ensemble of simulations.
"""
struct Ensemble{T<:Real}
    simlist::Vector{Simulation{T}}
    id::String
    datapath::String
    plotpath::String
end
function Ensemble(sl::Vector{Simulation{T}},id::String) where {T<:Real} 
    return Ensemble(sl,id,
                        "/home/how09898/phd/data/hhgjl/$id",
                        "/home/how09898/phd/plots/hhgjl/$id")
end
Ensemble(sl::Vector{Simulation{T}},id) where {T<:Real}      = Ensemble(sl,String(id)) 
Ensemble(sl::Vector{Simulation{T}}) where {T<:Real}         = Ensemble(sl,randstring(4)) 

Base.size(a::Ensemble)                  = (size(a.simlist))
Base.setindex!(a::Ensemble,v,i::Int)    = (a.simlist[i] = v)
Base.getindex(a::Ensemble,i::Int)       = a.simlist[i]
Base.length(a::Ensemble)                = length(a.simlist)
Base.firstindex(a::Ensemble)            = 1
Base.lastindex(a::Ensemble)             = length(a)

function Base.show(io::IO,::MIME"text/plain",e::Ensemble{T}) where {T}
    println(io,"Ensemble{$T} of $(length(e)) Simulations{$T}:")
    println(io," id: $(e.id)")
    println(io," datapath: $(e.datapath)")
    println(io," plotpath: $(e.plotpath)")
end

function getshortname(ens::Ensemble{T}) where {T<:Real}
    return "Ensemble[$(length(ens.simlist))]{$T}($(ens[1].dimensions)d)" * 
            getshortname(ens[1].hamiltonian) * getshortname(ens[1].drivingfield)
end

getname(ens::Ensemble) = getshortname(ens) * ens.id


function make_ensemble_from_path(
    path::String,
    ensembleid="default",
    datapath="/home/how09898/phd/data/hhgjl/default",
    plotpath="/home/how09898/phd/plots/hhgjl/default")
    
    simfiles = find_files_with_name(path,"simulation.meta")
    if isempty(simfiles)
        throw(ErrorException("No simulation.meta files found in path:\n$path"))
    end
    simlist  = load.(simfiles)

    return Ensemble(simlist,ensembleid,datapath,plotpath)
end


function parametersweep(
    sim::Simulation{T},
    comp::SimulationComponent{T},
    param::Symbol,
    range::AbstractVector{T};
    id="",
    plotpath="",
    datapath="") where {T<:Real}

    return parametersweep(sim,comp,[param],[(r,) for r in range];
        id=id,
        plotpath=plotpath,
        datapath=datapath)
end

function parametersweep(
    sim::Simulation{T},
    comp::SimulationComponent{T},
    params::Vector{Symbol},
    range::Vector{Tuple{Vararg{T, N}}};
    id="",
    plotpath="",
    datapath="") where {T<:Real,N}

    random_name  = "$(today())_" * random_word()
    plotpath     = plotpath == "" ? droplast(sim.plotpath) : plotpath
    datapath     = datapath == "" ? droplast(sim.datapath) : datapath
    ensname      = "Ensemble[$(length(range))]($(sim.dimensions)d)" 
    ensname      *= getshortname(sim.hamiltonian) *"_"* getshortname(sim.drivingfield) * "_"
    ensname      *= stringexpand_vector(params)*"_sweep_" * random_name
    id           = id == "" ? stringexpand_vector(params)*"_sweep_" * random_name : id
    

    sweeplist = Vector{Simulation{T}}(undef, length(range))
    for i in eachindex(sweeplist)

        name = ""
        for (p, v) in zip(params, range[i])
            name *= "$p=$(v)_"
        end
        name = name[1:end-1] # drop last underscore

        new_h = deepcopy(sim.hamiltonian)
        new_df = deepcopy(sim.drivingfield)
        new_p = deepcopy(sim.numericalparams)

        if comp isa Hamiltonian{T}
            for (p, v) in zip(params, range[i])
                new_h = set(new_h, PropertyLens(p), v)
            end
        elseif comp isa DrivingField{T}
            for (p, v) in zip(params, range[i])
                new_df = set(new_df, PropertyLens(p), v)
            end
        elseif comp isa NumericalParameters{T}
            for (p, v) in zip(params, range[i])
                new_p = set(new_p, PropertyLens(p), v)
            end
        end
        sweeplist[i] = Simulation(new_h, new_df, new_p, deepcopy(sim.observables),
            sim.unitscaling, sim.dimensions, name,
            joinpath(datapath, ensname, name * "/"),
            joinpath(plotpath, ensname, name * "/"))
    end


    return Ensemble(
        sweeplist,
        id,
        joinpath(datapath, ensname * "/"),
        joinpath(plotpath, ensname * "/"))
end


