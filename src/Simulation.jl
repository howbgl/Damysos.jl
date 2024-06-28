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
- `dimensions::UInt8`: system can be 0d (single mode),1d or 2d

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
    id::String
    datapath::String
    plotpath::String
    dimensions::UInt8
    function Simulation{T}(l,df,p,obs,us,id,dpath,ppath,d) where {T<:Real}
        if d != getdimension(p)
            @warn """
            The dimension d=$d does not match the the NumericalParameters.
            Overwriting to d=$(getdimension(p)) instead."""
        end
        new(l,df,p,obs,us,id,dpath,ppath,getdimension(p))
    end
end

function Simulation(
    l::Liouvillian{T},
    df::DrivingField{T},
    p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},
    us::UnitScaling{T},
    id::String,
    dpath::String,
    ppath::String,
    d::Integer=getdimension(p)) where {T<:Real} 

    return Simulation{T}(l,df,p,obs,us,id,dpath,ppath,UInt8(d))
end

function Simulation(
    l::Liouvillian{T},
    df::DrivingField{T},
    p::NumericalParameters{T},
    obs::Vector{O} where {O<:Observable{T}},
    us::UnitScaling{T},
    id::String) where {T<:Real}

    d    = getdimension(p)
    name = "Simulation{$T}($(d)d)" * getshortname(l) *"_"*  getshortname(df) * "_$id"
    return Simulation(
        l,
        df,
        p,
        obs,
        us,
        String(id),
        "/home/how09898/phd/data/hhgjl/"*name*"/",
        "/home/how09898/phd/plots/hhgjl/"*name*"/")
end

function Simulation(
    l::Liouvillian,
    df::DrivingField,
    p::NumericalParameters,
    obs::Vector{O} where {O<:Observable},
    us::UnitScaling)

    id = string(hash([l,df,p,obs,us]),base=16)
    return Simulation(l,df,p,obs,us,id)
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

function checkbzbounds(sim::Simulation)
    sim.numericalparams isa NumericalParamsSingleMode && return
    bz = getbzbounds(sim)
    if bz[1] > bz[2] || (sim.dimensions == 2 && bz[3] > bz[4])
        @warn "Brillouin zone vanishes: $(bz)"
    end
end

function resize_obs!(sim::Simulation)

    sim.observables .= [resize(o, sim.numericalparams) for o in sim.observables]
end

function buildkgrid_chunks(sim::Simulation,kchunksize::Integer)
    kxs            = collect(getkxsamples(sim.numericalparams))
    kys            = collect(getkysamples(sim.numericalparams))
    ks             = [getkgrid_point(i,kxs,kys) for i in 1:ntrajectories(sim)]
    return subdivide_vector(ks,kchunksize)
end

function define_functions(sim::Simulation,solver::DamysosSolver)
    !solver_compatible(sim,solver) && throw(incompatible_solver_exception(sim,solver))
    return (
        define_rhs_x(sim,solver),
        define_bzmask(sim,solver),
        define_observable_functions(sim,solver))
end

function incompatible_solver_exception(sim::Simulation,solver::DamysosSolver)
    return ErrorException("""
        Solver $solver is incompatible with simulation. Compatible pairs are:
            LinearChunked => 1d & 2d Simulation
            LinearCUDA    => 1d & 2d Simulation
            SingleMode    => 0d Simulation
        Your Simulation has the dimension $(sim.dimensions)""")
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

    str = """
        ζ = $ζ
        γ = $γ
        M = $M
        plz = $plz\n"""
    
    str *= printBZSI(sim.drivingfield,sim.numericalparams,sim.unitscaling,digits=digits)
    str *= printparamsSI(sim.liouvillian,sim.unitscaling;digits=digits)
    str *= printparamsSI(sim.drivingfield,sim.unitscaling;digits=digits)
    str *= printparamsSI(sim.numericalparams,sim.unitscaling;digits=digits)
    return str
end

printdimless_paramsSI(l::Liouvillian,df::DrivingField) = ""


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

