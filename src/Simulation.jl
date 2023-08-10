
struct UnitScaling{T<:Real} <: SimulationComponent{T}
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

function energySI(en,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"meV",en*Unitful.ħ/tc)
end
function electricfieldSI(field,us::UnitScaling)
    tc,lc   = getparams(us)
    e   = uconvert(u"C",1u"eV"/1u"V")
    return uconvert(u"MV/cm",field*Unitful.ħ/(e*tc*lc))
end
function timeSI(time,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"fs",time*tc)
end
function lengthSI(length,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å",length*lc)
end
function frequencySI(ν,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"THz",ν/tc)
end
function velocitySI(v,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"m/s",v*lc/tc)
end
function wavenumberSI(k,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å^-1",k/lc)
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
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 1"
            new(h,df,p,obs,us,UInt8(1),id,dpath,ppath)

        elseif p isa NumericalParams2d{T} && d!=2
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 2"
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
    name = "Simulation{$T}($(d)d)" * getshortname(h) *"_"*  getshortname(df) * "_$id"
    return Simulation(h,df,p,obs,us,d,String(id),
                "/home/how09898/phd/data/hhgjl/"*name*"/",
                "/home/how09898/phd/plots/hhgjl/"*name*"/")
end

function Simulation(h::Hamiltonian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer) where {T<:Real} 
    id = sprintf1("%x",hash([h,df,p,obs,us,d]))
    return Simulation(h,df,p,obs,us,d,id)
end

function Base.show(io::IO,::MIME"text/plain",s::Simulation{T}) where {T}
    print(io,"Simulation{$T} ($(s.dimensions)d) with components{$T}:\n")
    for n in fieldnames(Simulation{T})
        if !(n == :dimensions)
            if n == :observables
                println(io,"  Observables")
                str = ""
                for o in getfield(s,n)
                    str *= "    "*getshortname(o)*"\n"
                end 
                println(io,str)
            else
                print(io,"  ")
                Base.show(io,MIME"text/plain"(),getfield(s,n))
                print(io,'\n')
            end
        end
    end
end

function getshortname(sim::Simulation{T}) where {T<:Real}
    return "Simulation{$T}($(sim.dimensions)d)" * getshortname(sim.hamiltonian) *"_"* 
            getshortname(sim.drivingfield)
end

function getparams(sim::Simulation{T}) where {T<:Real}

    numpars     = getparams(sim.numericalparams)
    fieldpars   = getparams(sim.drivingfield)

    if sim.dimensions==1
        bztuple = (bz=(
            -numpars.kxmax + 1.3*fieldpars.eE/fieldpars.ω, 
            numpars.kxmax - 1.3*fieldpars.eE/fieldpars.ω
            ),)
    elseif sim.dimensions==2
        bztuple = (bz=(
            -numpars.kxmax + 1.3*fieldpars.eE/fieldpars.ω, 
            numpars.kxmax - 1.3*fieldpars.eE/fieldpars.ω,
            -numpars.kymax, 
            numpars.kymax
            ),)
    end

    merge(bztuple,
        getparams(sim.hamiltonian),
        fieldpars,
        numpars,
        getparams(sim.unitscaling))
end

getnames_obs(sim::Simulation{T}) where {T<:Real} = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation{T}) where {T<:Real} = vcat(arekresolved.(sim.observables)...)
getname(sim::Simulation{T}) where {T<:Real}      = getshortname(sim)*'_'*sim.id


getshortname(obs::Observable)           = split("$obs",'{')[1]
getshortname(c::SimulationComponent)    = split("$c",'{')[1]

function Base.show(io::IO,::MIME"text/plain",c::SimulationComponent{T}) where {T}
    println(io,getshortname(c))
    print(io,c |> getparams |> stringexpand_nt |> prepend_spaces)
end

function printparamsSI(sim::Simulation;digits=3)

    p   = getparams(sim)
    γ   = round(p.Δ*p.ω / p.eE,sigdigits=digits)        # Keldysh parameter
    M   = round(2*p.Δ / p.ω,sigdigits=digits)           # Multi-photon number
    ζ   = round(M/γ,sigdigits=digits)                    # My dimless asymptotic ζ
    plz = round(exp(-π*p.Δ^2 / p.eE),sigdigits=digits)  # Maximal LZ tunnel prob

    str = "ζ = $ζ\nγ = $γ\nM = $M\nplz = $plz\n"

    str *= printparamsSI(sim.hamiltonian,sim.unitscaling;digits=digits)
    str *= printparamsSI(sim.drivingfield,sim.unitscaling;digits=digits)
    str *= printparamsSI(sim.numericalparams,sim.unitscaling;digits=digits)
    return str
end
