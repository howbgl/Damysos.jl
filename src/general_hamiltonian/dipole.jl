
export dx_cc
export dx_cv
export dx_vc
export dx_vv
export dy_cc
export dy_cv
export dy_vc
export dy_vv

n2(h::SVector{3,<:Number})  = 2ϵ(h)*(ϵ(h) + h[3])
n2(h::GeneralTwoBand)       = :(2*$(ϵ(h)) * ($(ϵ(h)) + $(hz(h))))

function dμ_cv(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return -im*vμ_cv(h,dh) / Δϵ(h)
end
function dμ_cv(h::GeneralTwoBand,dh::SVector{3,<:Any})
    return :(-im * $(vμ_cv(h,dh)) / $(Δϵ(h)))
end

function dμ_vc(h::SVector{3,<:Number},dh::SVector{3,<:Number})
    return im*vμ_vc(h,dh) / Δϵ(h)
end
function dμ_vc(h::GeneralTwoBand,dh::SVector{3,<:Any})
    return :(im * $(vμ_vc(h,dh)) / $(Δϵ(h)))
end


function getdμ_cv(hvec::Function,dhdμ::Function)
    return let Δϵ_loc=getΔϵ(hvec)
        (kx,ky) -> -im*sum(σvec_cv(hvec(kx,ky)) .* dhdμ(kx,ky)) / Δϵ_loc(kx,ky)
    end
end

function getdμ_vc(hvec::Function,dhdμ::Function)
    return let Δϵ_loc=getΔϵ(hvec)
        (kx,ky) -> im*sum(σvec_vc(hvec(kx,ky)) .* dhdμ(kx,ky)) / Δϵ_loc(kx,ky)
    end
end

function getdμ_cc(hvec::Function,dhxdkμ::Function,dhydkμ::Function)
    return let Δϵ_loc=getΔϵ(hvec)
        (kx,kky) -> (dhxdkμ(kx,ky))
    end
end


function dx_cc(h::SVector{3,<:Number},jac::SMatrix{3,2,<:Number})
    return (jac[1,1]*h[2] - h[1]*jac[2,1]) / n2(h)
end
function dx_cc(h::GeneralTwoBand,jac::SMatrix{3,2,<:Any})
    hv = hvec(h)
    return :(($(jac[1,1]) * $(hv[2]) - $(hv[1]) * $(jac[2,1])) / $(n2(h)))
end

function dx_vv(h::SVector{3,<:Number},jac::SMatrix{3,2,<:Number})
    return -(jac[1,1]*h[2] - h[1]*jac[2,1]) / n2(h)
end
function dx_vv(h::GeneralTwoBand,jac::SMatrix{3,2,<:Any})
    hv = hvec(h)
    return :(-($(jac[1,1]) * $(hv[2]) - $(hv[1]) * $(jac[2,1])) / $(n2(h)))
end

function dy_cc(h::SVector{3,<:Number},jac::SMatrix{3,2,<:Number})
    return (jac[1,2]*h[2] - h[1]*jac[2,2]) / n2(h)
end
function dy_cc(h::GeneralTwoBand,jac::SMatrix{3,2,<:Any})
    hv = hvec(h)
    return :(($(jac[1,2]) * $(hv[2]) - $(hv[1]) * $(jac[2,2])) / $(n2(h)))    
end

function dy_vv(h::SVector{3,<:Number},jac::SMatrix{3,2,<:Number})
    return -(jac[1,2]*h[2] - h[1]*jac[2,2]) / n2(h)
end
function dy_vv(h::GeneralTwoBand,jac::SMatrix{3,2,<:Any})
    hv = hvec(h)
    return :(-($(jac[1,2]) * $(hv[2]) - $(hv[1]) * $(jac[2,2])) / $(n2(h)))
end

"""
    dx_cv(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ+|x|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dx_cv(h,1.0,-1.0)
-0.0610042339640731 - 0.22767090063073978im
```
"""
dx_cv(h::GeneralTwoBand,kx,ky)  = dμ_cv(hvec(h,kx,ky),dhdkx(h,kx,ky))
dx_cv(h::GeneralTwoBand)        = dμ_cv(h,dhdkx(h))

"""
    dx_vc(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ-|x|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dx_vc(h,1.0,-1.0)
-0.0610042339640731 + 0.22767090063073978im
```
"""
dx_vc(h::GeneralTwoBand,kx,ky)  = dμ_vc(hvec(h,kx,ky),dhdkx(h,kx,ky))
dx_vc(h::GeneralTwoBand)        = dμ_vc(h,dhdkx(h))

"""
    dx_cc(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ+|x|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dx_cc(h,1.0,-1.0)
-0.10566243270259357
```
"""
dx_cc(h::GeneralTwoBand,kx,ky)  = dx_cc(hvec(h,kx,ky),jac(h,kx,ky))
dx_cc(h::GeneralTwoBand)        = dx_cc(h,jac(h))

"""
    dx_vv(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ-|x|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dx_vv(h,1.0,-1.0)
0.10566243270259357
```
"""
dx_vv(h::GeneralTwoBand,kx,ky)  = dx_vv(hvec(h,kx,ky),jac(h,kx,ky))
dx_vv(h::GeneralTwoBand)        = dx_vv(h,jac(h))


"""
    dy_cv(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ+|y|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dy_cv(h,1.0,-1.0)
-0.22767090063073978 - 0.0610042339640731im
```
"""
dy_cv(h::GeneralTwoBand,kx,ky)  = dμ_cv(hvec(h,kx,ky),dhdky(h,kx,ky))
dy_cv(h::GeneralTwoBand)        = dμ_cv(h,dhdky(h))

"""
    dy_vc(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ-|y|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dy_vc(h,1.0,-1.0)
-0.22767090063073978 + 0.0610042339640731im
```
"""
dy_vc(h::GeneralTwoBand,kx,ky)  = dμ_vc(hvec(h,kx,ky),dhdky(h,kx,ky))
dy_vc(h::GeneralTwoBand)        = dμ_vc(h,dhdky(h))

"""
    dy_cc(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ+|y|ψ+⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dy_cc(h,1.0,-1.0)
-0.10566243270259357
```
"""
dy_cc(h::GeneralTwoBand,kx,ky)  = dy_cc(hvec(h,kx,ky),jac(h,kx,ky))
dy_cc(h::GeneralTwoBand)        = dy_cc(h,jac(h))

"""
    dy_vv(h,kx,ky)

Returns the dipole operator matrix element ⟨ψ-|y|ψ-⟩ at ``\\vec{k}=[k_x,k_y]``.

# Example
```jldoctest
julia> h = GappedDirac(1.0); dy_vv(h,1.0,-1.0)
0.10566243270259357
```
"""
dy_vv(h::GeneralTwoBand,kx,ky)  = dy_vv(hvec(h,kx,ky),jac(h,kx,ky))
dy_vv(h::GeneralTwoBand)        = dy_vv(h,jac(h))


