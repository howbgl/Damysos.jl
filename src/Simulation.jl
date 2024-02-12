export Simulation
export UnitScaling

export electricfieldSI
export energySI
export frequencySI
export getparams
export getparamsSI
export lengthSI
export printparamsSI
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
lengthscaleSI(us::UnitScaling)  = Quantity(us.lengthscale,u"nm")
timescaleSI(us::UnitScaling)    = Quantity(us.timescale,u"fs")
getparams(us::UnitScaling)      = (timescale=timescaleSI(us),lengthscale=lengthscaleSI(us))
getparamsSI(us::UnitScaling)    = (timescale=timescaleSI(us),lengthscale=lengthscaleSI(us))

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
    liouvillian::Liouvillian{T}
    drivingfield::DrivingField{T}
    numericalparams::NumericalParameters{T}
    observables::Vector{Observable{T}}
    unitscaling::UnitScaling{T}
    dimensions::UInt8
    id::String
    datapath::String
    plotpath::String
    function Simulation{T}(l,df,p,obs,us,d,id,dpath,ppath) where {T<:Real}

        if p isa NumericalParams1d{T} && d!=1
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 1"
            new(l,df,p,obs,us,UInt8(1),id,dpath,ppath)

        elseif p isa NumericalParams2d{T} && d!=2
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 2"
            new(l,df,p,obs,us,UInt8(2),id,dpath,ppath)
            
        else
            new(l,df,p,obs,us,d,id,dpath,ppath)
        end
    end
end

function Simulation(l::Liouvillian{T},df::DrivingField{T},p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},us::UnitScaling{T},d::Integer,
    id::String,dpath::String,ppath::String) where {T<:Real} 

    return Simulation{T}(l,df,p,obs,us,UInt8(abs(d)),id,dpath,ppath)
end

function Simulation(
    l::Liouvillian{T},
    df::DrivingField{T},
    p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},
    us::UnitScaling{T},
    d::Integer,
    id) where {T<:Real}

    name = "Simulation{$T}($(d)d)" * getshortname(l) *"_"*  getshortname(df) * "_$id"
    return Simulation(
        l,
        df,
        p,
        obs,
        us,
        d,
        String(id),
        "/home/how09898/phd/data/hhgjl/"*name*"/",
        "/home/how09898/phd/plots/hhgjl/"*name*"/")
end

function Simulation(
    l::Liouvillian,
    df::DrivingField,
    p::NumericalParameters,
    obs::Vector{O} where {O<:Observable},
    us::UnitScaling,
    d::Integer)

    id = sprintf1("%x",hash([l,df,p,obs,us,d]))
    return Simulation(l,df,p,obs,us,d,id)
end

function Base.show(io::IO,::MIME"text/plain",s::Simulation{T}) where T

    buf = IOBuffer()
    print(io,"Simulation{$T} ($(s.dimensions)d):\n")
    
    for n in fieldnames(Simulation{T})
        if !(n == :dimensions)
            if n == :observables
                println(io," Observables:")
                str = join([getshortname(o) for o in getfield(s,n)],"\n")
                println(io,prepend_spaces(str,2))
            elseif getfield(s,n) isa SimulationComponent
                Base.show(buf,MIME"text/plain"(),getfield(s,n))
                str = String(take!(buf))
                print(io,prepend_spaces(str)*"\n")
            else
                Base.show(buf,MIME"text/plain"(),getfield(s,n))
                str = String(take!(buf))
                println(io," $n: "*str)
            end
        end
    end
end

function getshortname(sim::Simulation)
    return "Simulation{$T}($(sim.dimensions)d)" * getshortname(sim.hamiltonian) *"_"* 
            getshortname(sim.drivingfield)
end

function getparams(sim::Simulation)

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
        getparams(sim.liouvillian),
        fieldpars,
        numpars,
        getparams(sim.unitscaling),
        (dimensions=sim.dimensions,))
end

function checkbzbounds(sim::Simulation)
    p = getparams(sim)
    if p.bz[1] > p.bz[2] || p.bz[3] > p.bz[4]
        @warn "Brillouin zone vanishes: $(p.bz)"
    end
end

getnames_obs(sim::Simulation)   = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation)   = vcat(arekresolved.(sim.observables)...)
getname(sim::Simulation)        = getshortname(sim)*'_'*sim.id


getshortname(obs::Observable)           = split("$obs",'{')[1]
getshortname(c::SimulationComponent)    = split("$c",'{')[1]

function Base.show(io::IO,::MIME"text/plain",c::Union{SimulationComponent,Hamiltonian})
    println(io,getshortname(c)*":")
    print(io,c |> getparams |> stringexpand_nt |> prepend_spaces)
end

function printparamsSI(sim::Simulation;digits=3)

    p   = getparams(sim)
    γ   = round(p.Δ*p.ω / p.eE,sigdigits=digits)        # Keldysh parameter
    M   = round(2*p.Δ / p.ω,sigdigits=digits)           # Multi-photon number
    ζ   = round(M/γ,sigdigits=digits)                   # My dimless asymptotic ζ
    plz = round(exp(-π*p.Δ^2 / p.eE),sigdigits=digits)  # Maximal LZ tunnel prob
    bzSI  = [wavenumberSI(k,sim.unitscaling) for k in p.bz]
    bzSI  = map(x -> round(typeof(x),x,sigdigits=digits),bzSI)
    bz    = [round(x,sigdigits=digits) for x in p.bz]

    str = """
        ζ = $ζ
        γ = $γ
        M = $M
        plz = $plz
        BZ(kx) = [$(bzSI[1]),$(bzSI[2])] ([$(bz[1]),$(bz[2])])
        BZ(ky) = [$(bzSI[3]),$(bzSI[4])] ([$(bz[3]),$(bz[4])])\n"""

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

