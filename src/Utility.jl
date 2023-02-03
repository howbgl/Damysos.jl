
function semiclassical_interband_range(h::GappedDirac,df::GaussianPulse)
    ϵ        = getϵ(h)
    ωmin     = 2.0*ϵ(0.0,0.0)
    kmax     = df.eE/df.ω
    ωmax     = 2.0*ϵ(kmax,0.0)
    min_harm = ωmin/df.ω
    max_harm = ωmax/df.ω
    println("Approximate range of semiclassical interband: ",min_harm," to ",
            max_harm," (harmonic number)")
end
