

struct UnitScaling{T<:Real}
    timescale::Unitful.Time{T}
    lengthscale::Unitful.Length{T}
end

struct Simulation{T<:Real}
    hamiltonian::Hamiltonian{T}
    drivingfield::DrivingField{T}
    numericalparams::NumericalParameters{T}
    observables::Vector{Observable{T}}
    unitscaling::UnitScaling{T}
    dimensions::UInt8
    name::String
    datapath::String
    plotpath::String
    function Simulation{T}(h,df,p,obs,us,d,name,dpath,ppath) where {T<:Real}
        if p isa NumericalParams1d{T} && d!=1
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(1),name,dpath,ppath)
        elseif p isa NumericalParams2d{T} && d!=2
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(2),name,dpath,ppath)
        else
            new(h,df,p,obs,us,d,name,dpath,ppath)
        end
    end
end
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,
    name::String,dpath::String,ppath::String) where {T<:Real} = Simulation{T}(h,df,p,obs,us,UInt8(abs(d)),name,dpath,ppath)
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,name) where {T<:Real} = Simulation(h,df,p,obs,us,d,String(name),"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")
Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer) where {T<:Real} = Simulation(h,df,p,obs,us,d,randstring(4),"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")

function Base.show(io::IO,::MIME"text/plain",s::Simulation{T}) where {T}
    print(io,"Simulation{$T} ($(s.dimensions)d) with components{$T}:\n")
    for n in fieldnames(Simulation{T})
        if !(n == :dimensions)
            print(io,"  ")
            Base.show(io,MIME"text/plain"(),getfield(s,n))
            print(io,'\n')
        end
    end
end

function getshortname(sim::Simulation{T}) where {T<:Real}
    return "Simulation{$T}($(sim.dimensions)d)" * split("_$(sim.hamiltonian)",'{')[1] * split("_$(sim.drivingfield)",'{')[1]
end

function saveparams(sim::Simulation{T}) where {T<:Real}
    if !isdir(sim.datapath * getfilename(sim))
        mkpath(sim.datapath * getfilename(sim))
    end
    CSV.write(sim.datapath * getfilename(sim) * "/params.csv",DataFrame([getparams(sim)]))
end

getparams(sim::Simulation{T}) where {T<:Real} = merge((bz=(-sim.numericalparams.kxmax + 1.3*sim.drivingfield.eE/sim.drivingfield.ω, sim.numericalparams.kxmax - 1.3*sim.drivingfield.eE/sim.drivingfield.ω),),
    getparams(sim.hamiltonian),getparams(sim.drivingfield),getparams(sim.numericalparams))
getnames_obs(sim::Simulation{T}) where {T<:Real} = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation{T}) where {T<:Real} = vcat(arekresolved.(sim.observables)...)
getfilename(sim::Simulation{T}) where {T<:Real}  = getshortname(sim)*'_'*sim.name


struct Ensemble{T<:Real}
    simlist::Vector{Simulation{T}}
    name::String
    datapath::String
    plotpath::String
end
Ensemble(sl::Vector{Simulation{T}},name::String) where {T<:Real} = Ensemble(sl,name,"/home/how09898/phd/data/hhgjl/","/home/how09898/phd/plots/hhgjl/")
Ensemble(sl::Vector{Simulation{T}},name) where {T<:Real}         = Ensemble(sl,String(name)) 
Ensemble(sl::Vector{Simulation{T}}) where {T<:Real}              = Ensemble(sl,"defaultens") 

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
    return "Ensemble{$T}[$(length(ens.simlist))]($(ens.simlist[1].dimensions)d)" * split("_$(ens.simlist[1].hamiltonian)",'{')[1] * split("_$(ens.simlist[1].drivingfield)",'{')[1]
end

getfilename(ens::Ensemble{T}) where {T<:Real} = getshortname(ens) * ens.name

getshortname(obs::Observable{T}) where {T<:Real} = split("$obs",'{')[1]

function parametersweep(sim::Simulation{T}, comp::SimulationComponent{T}, param::Symbol, range::AbstractVector{T}) where {T<:Real}
    sweeplist    = Vector{Simulation{T}}(undef,length(range))
    if comp isa Hamiltonian{T}
        for i in 1:length(sweeplist)
            new_h           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(new_h,sim.drivingfield,sim.numericalparams,sim.observables,sim.unitscaling,sim.dimensions)
        end
    elseif comp isa DrivingField{T}
        for i in 1:length(sweeplist)
            new_df          = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,new_df,sim.numericalparams,sim.observables,sim.unitscaling,sim.dimensions)
        end
    elseif comp isa NumericalParameters{T}
        for i in 1:length(sweeplist)
            new_p           = set(comp,PropertyLens(param),range[i])
            sweeplist[i]    = Simulation(sim.hamiltonian,sim.drivingfield,new_p,sim.observables,sim.unitscaling,sim.dimensions)
        end
    else
        return nothing
    end
    return Ensemble(sweeplist)
end

function Base.show(io::IO,::MIME"text/plain",c::SimulationComponent{T}) where {T}
    pars = getparams(c)
    print(io,split("$c",'{')[1],":  $pars")
end
