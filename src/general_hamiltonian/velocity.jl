
export getvx_cc
export getvx_cv
export getvx_vc
export getvx_vc
export getvx_vv
export getvy_cc
export getvy_cv
export getvy_vc
export getvy_vv
export vx_cc
export vx_cv
export vx_vc
export vx_vv
export vy_cc
export vy_cv
export vy_vc
export vy_vv


const VELOCITY_SYMBOLS = (:vx_cc,:vx_cv,:vx_vc,:vx_vv,:vy_cc,:vy_cv,:vy_vc,:vy_vv,
    :getvx_cc,:getvx_cv,:getvx_vc,:getvx_vv,:getvy_cc,:getvy_cv,:getvy_vc,:getvy_vv)


function vμ_cv(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return dh[1]*σx_cv(h) + dh[2]*σy_cv(h) + dh[3]*σz_cv(h) 
end
function vμ_cv(h::GeneralTwoBand,dh::SVector{3,<:Any})
    return :($(dh[1])*$(σx_cv(h)) + $(dh[2])*$(σy_cv(h)) + $(dh[3])*$(σz_cv(h))) 
end

function vμ_vc(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return dh[1]*σx_vc(h) + dh[2]*σy_vc(h) + dh[3]*σz_vc(h) 
end
function vμ_vc(h::GeneralTwoBand,dh::SVector{3,<:Any})
    return :($(dh[1])*$(σx_vc(h)) + $(dh[2])*$(σy_vc(h)) + $(dh[3])*$(σz_vc(h))) 
end

getvμ_cv(hvec::Function,dhdμ::Function) = (kx,ky) -> sum(σvec_cv(hvec(kx,ky)) .* dhdμ(kx,ky))
getvμ_vc(hvec::Function,dhdμ::Function) = (kx,ky) -> sum(σvec_vc(hvec(kx,ky)) .* dhdμ(kx,ky))


function vμ_cc(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return (h[1]*dh[1] + h[2]*dh[2] + h[3]*dh[3]) / ϵ(h)
end
function vμ_cc(h::GeneralTwoBand,dh::SVector{3,<:Any})
    hv = hvec(h)
    return :(($(hv[1])*$(dh[1]) + $(hv[2])*$(dh[2]) + $(hv[3])*$(dh[3])) / $(ϵ(h)))
end

function vμ_vv(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return -(h[1]*dh[1] + h[2]*dh[2] + h[3]*dh[3]) / ϵ(h)
end
function vμ_vv(h::GeneralTwoBand,dh::SVector{3,<:Any})
    hv = hvec(h)
    return :(-($(hv[1])*$(dh[1]) + $(hv[2])*$(dh[2]) + $(hv[3])*$(dh[3])) / $(ϵ(h)))
end

function getvμ_cc(hvec::Function,dhdμ::Function)
    return let ϵ_loc = getϵ(hvec)
        (kx,ky) -> sum(hvec(kx,ky) .* dhdμ(kx,ky)) / ϵ_loc(kx,ky)
    end
end
function getvμ_vv(hvec::Function,dhdμ::Function)
    return let ϵ_loc = getϵ(hvec)
        (kx,ky) -> -sum(hvec(kx,ky) .* dhdμ(kx,ky)) / ϵ_loc(kx,ky)
    end
end


"""
    vx_cv(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ+|vx|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vx_cv(h,1.0,-1.0)
0.7886751345948129 - 0.21132486540518708im
```
"""
vx_cv(h::GeneralTwoBand,kx,ky)  = vμ_cv(hvec(h,kx,ky),dhdkx(h,kx,ky))
vx_cv(h::GeneralTwoBand)        = vμ_cv(h,dhdkx(h))

"""
    vx_vc(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ-|vx|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vx_vc(h,1.0,-1.0)
0.7886751345948129 + 0.21132486540518708im
```
"""
vx_vc(h::GeneralTwoBand,kx,ky)  = vμ_vc(hvec(h,kx,ky),dhdkx(h,kx,ky))
vx_vc(h::GeneralTwoBand)        = vμ_vc(h,dhdkx(h))

"""
    vx_cc(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ+|vx|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vx_cc(h,1.0,-1.0)
0.5773502691896258
```
"""
vx_cc(h::GeneralTwoBand,kx,ky)  = vμ_cc(hvec(h,kx,ky),dhdkx(h,kx,ky))
vx_cc(h::GeneralTwoBand)        = vμ_cc(h,dhdkx(h))
"""
    vx_vv(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ-|vx|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vx_vv(h,1.0,-1.0)
-0.5773502691896258
```
"""
vx_vv(h::GeneralTwoBand,kx,ky)  = vμ_vv(hvec(h,kx,ky),dhdkx(h,kx,ky))
vx_vv(h::GeneralTwoBand)        = vμ_vv(h,dhdkx(h))

"""
    vy_cv(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ+|vy|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vy_cv(h,1.0,-1.0)
0.21132486540518708 - 0.7886751345948129im
```
"""
vy_cv(h::GeneralTwoBand,kx,ky)  = vμ_cv(hvec(h,kx,ky),dhdky(h,kx,ky))
vy_cv(h::GeneralTwoBand)        = vμ_cv(h,dhdky(h))

"""
    vy_vc(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ-|vy|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vy_vc(h,1.0,-1.0)
0.21132486540518708 + 0.7886751345948129im
```
"""
vy_vc(h::GeneralTwoBand,kx,ky)  = vμ_vc(hvec(h,kx,ky),dhdky(h,kx,ky))
vy_vc(h::GeneralTwoBand)        = vμ_vc(h,dhdky(h ))

"""
    vy_cc(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ+|vy|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vy_cc(h,1.0,-1.0)
-0.5773502691896258
```
"""
vy_cc(h::GeneralTwoBand,kx,ky)  = vμ_cc(hvec(h,kx,ky),dhdky(h,kx,ky))
vy_cc(h::GeneralTwoBand)        = vμ_cc(h,dhdky(h))

"""
    vy_vv(h,kx,ky)

Returns the velocity operator matrix element ⟨ψ-|vy|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); vy_vv(h,1.0,-1.0)
0.5773502691896258
```
"""
vy_vv(h::GeneralTwoBand,kx,ky)  = vμ_vv(hvec(h,kx,ky),dhdky(h,kx,ky))
vy_vv(h::GeneralTwoBand)        = vμ_vv(h,dhdky(h))


getvx_cc(h::GeneralTwoBand) = getvμ_cc(gethvec(h),getdhdkx(h))
getvx_cv(h::GeneralTwoBand) = getvμ_cv(gethvec(h),getdhdkx(h))
getvx_vc(h::GeneralTwoBand) = getvμ_vc(gethvec(h),getdhdkx(h))
getvx_vv(h::GeneralTwoBand) = getvμ_vv(gethvec(h),getdhdkx(h))
getvy_cc(h::GeneralTwoBand) = getvμ_cc(gethvec(h),getdhdky(h))
getvy_cv(h::GeneralTwoBand) = getvμ_cv(gethvec(h),getdhdky(h))
getvy_vc(h::GeneralTwoBand) = getvμ_vc(gethvec(h),getdhdky(h))
getvy_vv(h::GeneralTwoBand) = getvμ_vv(gethvec(h),getdhdky(h))
