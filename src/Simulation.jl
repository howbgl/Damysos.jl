
struct UnitScaling{T<:Real}
    timescale::T
    lengthscale::T
end
function UnitScaling(timescale,lengthscale) 
    return UnitScaling(ustrip(u"fs",timescale),ustrip(u"nm",lengthscale))
end
function getparams(us::UnitScaling{T}) where {T<:Real} 
    return (timescale=Quantity(us.timescale,u"fs"),
            lengthscale=Quantity(us.lengthscale,u"nm"))
end


struct Simulation{T<:Real}
    hamiltonian::Hamiltonian{T}
    drivingfield::DrivingField{T}
    numericalparams::NumericalParameters{T}
    observables::Vector{Observable{T}}
    unitscaling::UnitScaling{T}
    dimensions::UInt8
    id::String
    datapath::String
    plotpath::String
    function Simulation{T}(h,df,p,obs,us,d,id,dpath,ppath) where {T<:Real}
        if p isa NumericalParams1d{T} && d!=1
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(1),id,dpath,ppath)
        elseif p isa NumericalParams2d{T} && d!=2
            printstyled("warning: ",color = :red)
            print("given dimensions ($d) not matching $p")
            println("; setting dimensions to 1")
            new(h,df,p,obs,us,UInt8(2),id,dpath,ppath)
        else
            new(h,df,p,obs,us,d,id,dpath,ppath)
        end
    end
end

function Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,
    id::String,dpath::String,ppath::String) where {T<:Real} 

    return Simulation{T}(h,df,p,obs,us,UInt8(abs(d)),id,dpath,ppath)
end

function Simulation(h::Hamiltonian{T},df::DrivingField{T},
    p::NumericalParameters{T},obs::Vector{O} where {O<:Observable{T}},
    us::UnitScaling{T},d::Integer,id) where {T<:Real} 

    return Simulation(h,df,p,obs,us,d,String(id),
                "/home/how09898/phd/data/hhgjl/",
                "/home/how09898/phd/plots/hhgjl/")
end

function Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer) where {T<:Real} 
    
    return Simulation(h,df,p,obs,us,d,randstring(4),
                "/home/how09898/phd/data/hhgjl/",
                "/home/how09898/phd/plots/hhgjl/")
end

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
    return "Simulation{$T}($(sim.dimensions)d)" * getshortname(sim.hamiltonian) * 
            getshortname(sim.drivingfield)
end

function getparams(sim::Simulation{T}) where {T<:Real}
    merge((bz=(-sim.numericalparams.kxmax + 1.3*sim.drivingfield.eE/sim.drivingfield.ω, 
        sim.numericalparams.kxmax - 1.3*sim.drivingfield.eE/sim.drivingfield.ω),),
        getparams(sim.hamiltonian),
        getparams(sim.drivingfield),
        getparams(sim.numericalparams),
        getparams(sim.unitscaling))
end

getnames_obs(sim::Simulation{T}) where {T<:Real} = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation{T}) where {T<:Real} = vcat(arekresolved.(sim.observables)...)
getname(sim::Simulation{T}) where {T<:Real}      = getshortname(sim)*'_'*sim.id


getshortname(obs::Observable{T}) where {T<:Real} = split("$obs",'{')[1]

function Base.show(io::IO,::MIME"text/plain",c::SimulationComponent{T}) where {T}
    pars = getparams(c)
    print(io,split("$c",'{')[1],":  $pars")
end
