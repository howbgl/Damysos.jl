
function printdimless_params(l::TwoBandDephasingLiouvillian,df::DrivingField;digits=3)
    return printdimless_params(l.hamiltonian,df;digits=digits)    
end

function printdimless_params(h::GeneralTwoBand,df::DrivingField;digits=3)
    return "No simple dimensionless parameters defined for this combination " *
        "of hamiltonian and driving field.\n\n" 
end

function printdimless_params(h::GappedDirac,df::Union{GaussianAPulse,GaussianEPulse};
    digits=3)

    amax = maximum_vecpot(df)
    emax = maximum_efield(df)
    ω   = central_angular_frequency(df)
    γ   = round(h.m/ (2amax),sigdigits=digits)          # Keldysh parameter
    M   = round(2h.m / ω,sigdigits=digits)            # Multi-photon number
    ζ   = round(M/2γ,sigdigits=digits)                   # My dimless asymptotic ζ
    plz = round(exp(-π*h.m^2 / emax),sigdigits=digits)  # Maximal LZ tunnel prob

    return """
        ζ = $ζ
        γ = $γ
        M = $M
        plz = $plz\n"""
end

function printdimless_params(
    h::Union{QuadraticToy,BilayerToy},
    df::Union{GaussianAPulse,GaussianEPulse};
    digits=3)

    emax = maximum_efield(df)
    ω   = central_angular_frequency(df)
    M   = round(h.Δ / ω,sigdigits=digits)               # Multi-photon number
    ζ   = round(h.ζ * emax^2  / ω^3,sigdigits=digits)   # My dimless asymptotic ζ

    # In SI units: ζ = (eE)^2 / (ħω * m)

    return """
        M = $M
        ζ = $ζ"""
end