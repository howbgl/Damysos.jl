
struct Ensemble{T<:Real}
    simlist::Vector{Simulation{T}}
    name::String
    datapath::String
    plotpath::String
end
function Ensemble(sl::Vector{Simulation{T}},name::String) where {T<:Real} 
    return Ensemble(sl,name,
                        "/home/how09898/phd/data/hhgjl/",
                        "/home/how09898/phd/plots/hhgjl/")
end
Ensemble(sl::Vector{Simulation{T}},name) where {T<:Real}    = Ensemble(sl,String(name)) 
Ensemble(sl::Vector{Simulation{T}}) where {T<:Real}         = Ensemble(sl,"defaultens") 

Base.size(a::Ensemble)                  = (size(a.simlist))
Base.setindex!(a::Ensemble,v,i::Int)    = (a.simlist[i] = v)
Base.getindex(a::Ensemble,i::Int)       = a.simlist[i]
function Base.show(io::IO,::MIME"text/plain",e::Ensemble{T}) where {T}
    print(io,"Ensemble{$T} of Simulations{$T}:\n")
    for i in 1:length(e.simlist)
        print(io,"  #$i\n","  ")
        Base.show(io,MIME"text/plain"(),e.simlist[i])
        print(io,"\n")
    end
end

function getshortname(ens::Ensemble{T}) where {T<:Real}
    return "Ensemble[$(length(ens.simlist))]{$T}($(ens[1].dimensions)d)" * 
            getshortname(ens[1].hamiltonian) * getshortname(ens[1].drivingfield)
end

getname(ens::Ensemble{T}) where {T<:Real} = getshortname(ens) * ens.name
