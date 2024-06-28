
# Fallback methods by brute force, more specialized methods are more efficient!
function getbzbounds(df::DrivingField,p::NumericalParams1d)
    ax      = get_vecpotx(df)
    ts      = gettsamples(p)
    axmax   = maximum(abs.(ax.(ts)))
    kxmax   = maximum(getkxsamples(p))
    return (-kxmax + 1.3axmax,kxmax - 1.3axmax)
end

function getbzbounds(df::DrivingField,p::NumericalParams2d)
    bz_1d = getbzbounds(df,NumericalParams1d(p.dkx,p.kxmax,0.0,p.dt,p.t0,p.rtol,p.atol))
    ay      = get_vecpoty(df)
    aymax   = maximum(abs.(ay.(ts)))
    kymax   = maximum(getkysamples(p))
    return (bz_1d...,-kymax + 1.3aymax,kymax - 1.3aymax)
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
