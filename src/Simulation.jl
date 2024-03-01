export Simulation
export UnitScaling

export electricfield_scaled
export electricfieldSI
export energyscaled
export energySI
export frequencyscaled
export frequencySI
export getparams
export lengthscaled
export lengthSI
export timescaled
export timeSI
export velocityscaled
export velocitySI
export wavenumberscaled
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

function energySI(en::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"meV",en*Unitful.ħ/tc)
end

function energyscaled(energy::Unitful.Energy,us::UnitScaling)
    tc,lc = getparams(us)
    ħ     = Unitful.ħ
    return uconvert(Unitful.NoUnits,tc*energy/ħ)
end

function electricfieldSI(field::Real,us::UnitScaling)
    tc,lc   = getparams(us)
    e       = uconvert(u"C",1u"eV"/1u"V")
    ħ       = Unitful.ħ
    return uconvert(u"MV/cm",field*ħ/(e*tc*lc))
end

function electricfield_scaled(field::Unitful.EField,us::UnitScaling)
    tc,lc   = getparams(us)
    e       = uconvert(u"C",1u"eV"/1u"V")
    ħ       = Unitful.ħ
    return uconvert(Unitful.NoUnits,e*tc*lc*field/ħ)
end

function timeSI(time::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"fs",time*tc)
end

function timescaled(time::Unitful.Time,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,time/tc)    
end

function lengthSI(length::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å",length*lc)
end
function lengthscaled(length::Unitful.Length,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,length/lc)
end

function frequencySI(ν::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"THz",ν/tc)
end

function frequencyscaled(ν::Unitful.Frequency,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,ν*tc)
end

function velocitySI(v::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"m/s",v*lc/tc)
end

function velocityscaled(v::Unitful.Velocity,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,v*tc/lc)
end

function wavenumberSI(k::Real,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(u"Å^-1",k/lc)
end

function wavenumberscaled(k::Unitful.Wavenumber,us::UnitScaling)
    tc,lc = getparams(us)
    return uconvert(Unitful.NoUnits,k*lc)
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


function Base.isapprox(
    s1::Simulation{T},
    s2::Simulation{U};
    atol::Real=0,
    rtol=atol>0 ? 0 : √eps(promote_type(T,U)),
    nans::Bool=false) where {T,U}
    
    coll = zip(s1.observables,s2.observables)
    return all([Base.isapprox(o1,o2;atol=atol,rtol=rtol,nans=nans) for (o1,o2) in coll])
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
        getparams(sim.unitscaling),
        (dimensions=sim.dimensions,))
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

