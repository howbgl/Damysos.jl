
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

function getbzbounds(df::GaussianAPulse,p::NumericalParams1d)
    kxmax = p.kxmax
    axmax = df.eE / df.ω
    return (-kxmax + 1.3axmax,kxmax - 1.3axmax)
end

function getbzbounds(df::GaussianAPulse,p::NumericalParams2d)

    amax = 1.3df.eE / df.ω
    return (
        -p.kxmax + cos(df.φ)*amax,
        p.kxmax - cos(df.φ)*amax,
        -p.kymax + sin(df.φ)*amax,
        p.kymax - sin(df.φ)*amax)
end

function checkbzbounds(sim::Simulation)
    bz = getbzbounds(sim)
    if bz[1] > bz[2] || bz[3] > bz[4]
        @warn "Brillouin zone vanishes: $(bz)"
    end
end
