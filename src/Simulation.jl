export Simulation
export UnitScaling

export electricfieldSI
export energySI
export frequencySI
export getparams
export lengthSI
export timeSI
export velocitySI
export wavenumberSI

"""
    UnitScaling(timescale,lengthscale)

Represents a physical length- and time-scale used for non-dimensionalization of a system.

# Examples
```jldoctest
julia> using Unitful; us = UnitScaling(u"1.0s",u"1.0m")
UnitScaling{Float64}(1.0e15, 1.0e9)
```

# Further information
See [here](https://en.wikipedia.org/w/index.php?title=Nondimensionalization&oldid=1166582079)
"""
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

"""
    Simulation{T}(hamiltonian, drivingfield, numericalparams, observables, unitscaling, dimensions, id, datapath, plotpath)

A struct representing a simulation with various components.
"""
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
    bztuple     = (bz=getbzbounds(sim),)

    merge(bztuple,
        getparams(sim.hamiltonian),
        fieldpars,
        numpars,
        getparams(sim.unitscaling),
        (dimensions=sim.dimensions,))
end

function getbzbounds(sim::Simulation)

    max_vecpot  = getmax_vecpot(sim)
    p           = sim.numericalparams
    bz          = (-p.kxmax + 1.3max_vecpot[1],p.kxmax - 1.3max_vecpot[1])
    if sim.dimensions==2
        bz = (bz...,-p.kymax + 1.3max_vecpot[2],p.kymax - 1.3max_vecpot[2])
    end
    return bz
end

getmax_vecpot(sim::Simulation) = [getmax_vecpot_x(sim),getmax_vecpot_y(sim)]

function getmax_vecpot_x(sim::Simulation)

    ax = get_vecpotx(sim.drivingfield)
    ts = gettsamples(sim.numericalparams)
    return maximum(ax.(ts))
end

function getmax_vecpot_y(sim::Simulation)

    ay = get_vecpoty(sim.drivingfield)
    ts = gettsamples(sim.numericalparams)
    return maximum(ay.(ts))
end

function checkbzbounds(sim::Simulation)
    p = getparams(sim)
    if p.bz[1] > p.bz[2] || p.bz[3] > p.bz[4]
        @warn "Brillouin zone vanishes: $(p.bz)"
    end
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

export printparamsSI
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


function markdown_paramsSI(sim::Simulation)

    input_str = printparamsSI(sim)
    table_str = "| Parameter | Value (SI units) | Value (scaled) |\n"*
                "|-----------|------------------|----------------|\n"

    # Split the input string into lines
    lines = split(input_str, '\n')

    for line in lines
        # Use regular expressions to extract values
        pattern = r"(.+?)\s*=\s*([^()]+)\s*(?:\(([\d\.]+)\))?"

        # Match the pattern in the input string
        match_result = match(pattern, line)

        if match_result !== nothing
            # Extract matched groups
            parameter_name = match_result[1]
            first_number = match_result[2]
            number_in_brackets = isnothing(match_result[3]) ? " " : match_result[3]

            # Append a new row to the table string
            table_str *= "| $parameter_name | $first_number | $number_in_brackets |\n"
        end
    end

    return table_str
end

