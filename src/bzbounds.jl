

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

function maximum_kdisplacement(df::DrivingField,ts::AbstractVector{<:Real})
    axmax = maximum(abs.(map(t -> vecpotx(df,t),ts)))
    aymax = maximum(abs.(map(t -> vecpoty(df,t),ts)))
    return maximum((axmax,aymax))
end