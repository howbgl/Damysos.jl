

struct GappedDirac{T<:Real} <: Hamiltonian{T}
    Δ::T
    t2::T
    GappedDirac{T}(Δ,t2) where {T<:Real} = new(Δ,t2) 
end
GappedDirac(Δ::T,t2::T) where {T<:Real}      = GappedDirac{T}(Δ,t2)
GappedDirac(Δ::Real,t2::Real)                = GappedDirac(promote(Δ,t2)...)
function GappedDirac(mass::Unitful.Energy{T},fermivelocity::Unitful.Velocity{T},
                    dephasing::Unitful.Time{T}) where{T<:Real}
    tc = uconvert(u"fs",0.1*Unitful.ħ/mass)
    lc = uconvert(u"nm",fermivelocity*tc)
    us = UnitScaling(tc,lc)
    return GappedDirac(us,mass,fermivelocity,dephasing)
end
GappedDirac(umass,ufermivelocity,udephasingtime)=GappedDirac(promote(umass,ufermivelocity,udephasingtime)...)
function GappedDirac(us::UnitScaling{T},mass::Unitful.Energy{T},
    fermivelocity::Unitful.Velocity{T},dephasing::Unitful.Time{T}) where{T<:Real}
    p   = getparams(us)
    Δ   = uconvert(Unitful.NoUnits,mass*p.timescale/Unitful.ħ)
    t2  = uconvert(Unitful.NoUnits,dephasing/p.timescale)
    return us,GappedDirac(Δ,t2)
end

getparams(h::GappedDirac{T}) where {T<:Real} = (Δ=h.Δ,t2=h.t2)

getϵ(h::GappedDirac{T})     where {T<:Real}  = (kx,ky) -> sqrt(kx^2+ky^2+h.Δ^2)
getdx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> ky * (1.0 -h. h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)
getdx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx + 2.0im*ky)
getdx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (ky/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*kx*h.Δ / (kx^2+ky^2+h.Δ^2)) / (2.0kx - 2.0im*ky)
getdx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -ky * (1.0 -h.Δ/sqrt(kx^2+ky^2+h.Δ^2)) / (2.0kx^2 + 2.0ky^2)

getvx_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> kx/sqrt(kx^2+ky^2+h.Δ^2)
getvx_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*ky) / (kx + 1.0im*ky)
getvx_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*kx/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*ky) / (kx - 1.0im*ky)
getvx_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -kx/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> ky/sqrt(kx^2+ky^2+h.Δ^2)
getvy_cv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) - 1.0im*kx) / (kx + 1.0im*ky)
getvy_vc(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> (h.Δ*ky/sqrt(kx^2+ky^2+h.Δ^2) + 1.0im*kx) / (kx - 1.0im*ky)
getvy_vv(h::GappedDirac{T}) where {T<:Real}  = (kx,ky) -> -ky/sqrt(kx^2+ky^2+h.Δ^2)

getdipoles_x(h::GappedDirac{T}) where {T<:Real}  = (getdx_cc(h),getdx_cv(h),getdx_vc(h),getdx_vv(h))
getvels_x(h::GappedDirac{T}) where {T<:Real}     = (getvx_cc(h),getvx_cv(h),getvx_vc(h),getvx_vv(h))  

