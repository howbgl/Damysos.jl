export Simulation


"""
    Simulation{T}(l, df, p, obs, us, d[, id, datapath, plotpath])

Represents a simulation with all physical and numerical parameters specified.

# Fields
- `l::Liouvillian{T}`: describes physical system via Liouville operator
- `df::DrivingField{T}`: laser field driving the system
- `p::NumericalParameters{T}`: all numerical parameters of the system
- `obs::Vector{Observable{T}}`: physical observables to be computed
- `us::UnitScaling{T}`: time- and lengthscale linking dimensionless units to SI units
- `id::String`: identifier of the Simulation
- `datapath::String`: path to save computed observables and simulation metadata
- `plotpath::String`: path to savee automatically generated plots

# See also
[`Ensemble`](@ref), [`TwoBandDephasingLiouvillian`](@ref), [`UnitScaling`](@ref),
[`Velocity`](@ref), [`Occupation`](@ref), [`GaussianAPulse`](@ref)
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

        _d = d
        if p isa NumericalParams1d{T} && d!=1
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 1"
            _d = 1
        elseif p isa NumericalParams2d{T} && d!=2
            @warn "given dimensions ($d) not matching $p\nsetting dimensions to 2"
            _d = 2
        end

        new(l,df,p,obs,us,_d,id,dpath,ppath)
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

    id = string(hash([l,df,p,obs,us,d]),base=16)
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

for func = (BAND_SYMBOLS...,DIPOLE_SYMBOLS...,VELOCITY_SYMBOLS...)
    @eval(Damysos,$func(s::Simulation) = $func(s.liouvillian))
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
    return "Simulation{$T}($(sim.dimensions)d)" * getshortname(sim.liouvillian) *"_"* 
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

    merge((bz=getbzbounds(sim),),
        getparams(sim.liouvillian),
        fieldpars,
        numpars,
        getparams(sim.unitscaling),
        (dimensions=sim.dimensions,))
end

getnames_obs(sim::Simulation)   = vcat(getnames_obs.(sim.observables)...)
arekresolved(sim::Simulation)   = vcat(arekresolved.(sim.observables)...)
getname(sim::Simulation)        = getshortname(sim)*'_'*sim.id


getshortname(obs::Observable)           = split("$obs",'{')[1]
getshortname(c::SimulationComponent)    = split("$c",'{')[1]


getbzbounds(sim::Simulation) = getbzbounds(sim.drivingfield,sim.numericalparams)
function getbzbounds(df::DrivingField,p::NumericalParameters)
    
    # Fallback method by brute force, more specialized methods are more efficient!
    ax      = get_vecpotx(df)
    ts      = gettsamples(p)
    axmax   = maximum(abs.(ax.(ts)))
    kxmax   = maximum(getkxsamples(p))
    
    bztuple = (-kxmax + 1.3axmax,kxmax - 1.3axmax)
    if sim.dimensions==2
        ay      = get_vecpoty(df)
        aymax   = maximum(abs.(ay.(ts)))
        kymax   = maximum(getkysamples(p))
        bztuple = (bztuple...,-kymax + 1.3aymax,kymax - 1.3aymax)
    end
    return bztuple
end


function checkbzbounds(sim::Simulation)
    bz = getbzbounds(sim)
    if bz[1] > bz[2] || bz[3] > bz[4]
        @warn "Brillouin zone vanishes: $(bz)"
    end
end

function resize_obs!(sim::Simulation{T}) where {T<:Real}

    sim.observables .= [resize(o, sim.numericalparams) for o in sim.observables]
end


function Base.show(io::IO,::MIME"text/plain",c::Union{SimulationComponent,Hamiltonian})
    println(io,getshortname(c)*":")
    print(io,c |> getparams |> stringexpand_nt |> prepend_spaces)
end

function printparamsSI(sim::Simulation;digits=3)

    p   = getparams(sim)
    γ   = round(p.m*p.ω / p.eE,sigdigits=digits)        # Keldysh parameter
    M   = round(2*p.m / p.ω,sigdigits=digits)           # Multi-photon number
    ζ   = round(M/γ,sigdigits=digits)                   # My dimless asymptotic ζ
    plz = round(exp(-π*p.m^2 / p.eE),sigdigits=digits)  # Maximal LZ tunnel prob
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

    str *= printparamsSI(sim.liouvillian,sim.unitscaling;digits=digits)
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

