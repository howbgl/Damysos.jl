
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
    simlist  = load.(simfiles)

    return Ensemble(simlist,ensembleid,datapath,plotpath)
end