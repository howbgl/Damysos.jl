struct GappedDirac{T<:Real} <: Hamiltonian{T}
    Δ::T
    t1::T
    t2::T
end
GappedDirac(Δ::Real,t1::Real,t2::Real)      = GappedDirac(promote(Δ,t1,t2)...)
GappedDirac(Δ::Real,t1::Real)               = GappedDirac(Δ,t1,Inf)
function GappedDirac(us::UnitScaling{T},mass::Unitful.Energy{T},
        fermivelocity::Unitful.Velocity{T},dephasing1::Unitful.Time{T},
        dephasing2::Unitful.Time{T}) where{T<:Real}
    p   = getparams(us)
    Δ   = uconvert(Unitful.NoUnits,mass*p.timescale/Unitful.ħ)
    t1  = uconvert(Unitful.NoUnits,dephasing1/p.timescale)
    t2  = uconvert(Unitful.NoUnits,dephasing2/p.timescale)
    return GappedDirac(Δ,t1,t2)
end
function GappedDirac(us::UnitScaling{T},mass::Unitful.Energy{T},
    fermivelocity::Unitful.Velocity{T},dephasing2::Unitful.Time{T}) where{T<:Real}
    return GappedDirac(us,mass,fermivelocity,zero(T),dephasing2)
end

function scalegapped_dirac(mass::Unitful.Energy{T},fermivelocity::Unitful.Velocity{T},
                    dephasing1::Unitful.Time{T},dephasing2::Unitful.Time{T};
                    tc_factor=0.1) where{T<:Real}
    mul = convert(T,tc_factor)
    tc = uconvert(u"fs",mul*Unitful.ħ/mass)
    lc = uconvert(u"nm",fermivelocity*tc)
    us = UnitScaling(tc,lc)
    return us,GappedDirac(us,mass,fermivelocity,dephasing1,dephasing2)
end
function scalegapped_dirac(umass,ufermivelocity,udephasingtime;tc_factor=0.1)
    return scalegapped_dirac(umass,ufermivelocity,Unitful.Quantity(Inf,u"s"),udephasingtime,
                            tc_factor=tc_factor)
end
function scalegapped_dirac(umass,ufermivelocity,udephasingtime1,udephasingtime2)
    return scalegapped_dirac(promote(umass,ufermivelocity,udephasingtime1,udephasingtime2)...;
                            tc_factor=tc_factor)
end

getparams(h::GappedDirac) = (Δ=h.Δ,t1=h.t1,t2=h.t2,vF=one(h.Δ))

getϵ(h::GappedDirac)     = (kx,ky) -> sqrt(kx^2+ky^2+h.Δ^2)
getdx_cc(h::GappedDirac) = (kx,ky) -> ky * (1-h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / 2(kx^2 + ky^2)
getdx_cv(h::GappedDirac) = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) - im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / 2(kx+im*ky)
getdx_vc(h::GappedDirac) = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) + im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / 2(kx-im*ky)
getdx_vv(h::GappedDirac) = (kx,ky) -> -ky*(1-h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / 2(kx^2 + ky^2)

getvx_cc(h::GappedDirac) = (kx,ky) -> kx/sqrt(kx^2+ky^2+h.Δ^2)
getvx_cv(h::GappedDirac) = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) + im*ky) / (kx + im*ky)
getvx_vc(h::GappedDirac) = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) - im*ky) / (kx - im*ky)
getvx_vv(h::GappedDirac) = (kx,ky) -> -kx/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cc(h::GappedDirac) = (kx,ky) -> ky/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cv(h::GappedDirac) = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) - im*kx) / (kx + im*ky)
getvy_vc(h::GappedDirac) = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) + im*kx) / (kx - im*ky)
getvy_vv(h::GappedDirac) = (kx,ky) -> -ky/sqrt(kx^2+ky^2+h.Δ^2)

getΔϵ(h::GappedDirac)    = (kx,ky) -> 2*sqrt(kx^2+ky^2+h.Δ^2)
getΔvx(h::GappedDirac)   = (kx,ky) -> 2*kx/sqrt(kx^2+ky^2+h.Δ^2)
getΔvy(h::GappedDirac)   = (kx,ky) -> 2*ky/sqrt(kx^2+ky^2+h.Δ^2)
getΔv(h::GappedDirac)    = (getΔvx(h),getΔvy(h))

getgxx(h::GappedDirac)   = (kx,ky) -> 2(h.Δ^2+ky^2) / (h.Δ^2+kx^2+ky^2)^(3/2)
getgxy(h::GappedDirac)   = (kx,ky) -> 2(-kx*ky) / (h.Δ^2+kx^2+ky^2)^(3/2)
getgyx(h::GappedDirac)   = getxy(h)
getgyy(h::GappedDirac)   = (kx,ky) -> 2(h.Δ^2+kx^2) / (h.Δ^2+kx^2+ky^2)^(3/2)


getdipoles_x(h::GappedDirac) = (getdx_cc(h),getdx_cv(h),getdx_vc(h),getdx_vv(h))
getvels_x(h::GappedDirac)    = (getvx_cc(h),getvx_cv(h),getvx_vc(h),getvx_vv(h))
getvels_y(h::GappedDirac)    = (getvy_cc(h),getvy_cv(h),getvy_vc(h),getvy_vv(h))    

function printparamsSI(h::GappedDirac,us::UnitScaling;digits=3)

    p   = getparams(h)
    Δ   = energySI(p.Δ,us)
    t1  = timeSI(p.t1,us)
    t2  = timeSI(p.t2,us)
    vF  = velocitySI(p.vF,us)

    symbols     = [:Δ,:t1,:t2,:vF]
    valuesSI    = [Δ,t1,t2,vF]
    values      = [getproperty(p,s) for s in symbols]

    str = ""

    for (s,v,vsi) in zip(symbols,values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return str
end
