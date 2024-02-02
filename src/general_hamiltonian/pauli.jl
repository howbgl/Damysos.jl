
export σvec_cv
export σvec_vc
export σx_cv
export σx_vc
export σy_cv
export σy_vc
export σz_cv
export σz_vc

σx_cv(h::GeneralTwoBand,kx,ky)          = σx_cv(hx(h,kx,ky),hy(h,kx,ky),hz(h,kx,ky))
σx_cv(hx::Number,hy::Number,hz::Number) = (im * hy + hx*hz/ϵ(hx,hy,hz)) / (hx + im*hy)
σx_cv(h::SVector{3,<:Number})           = (im * h[2] + h[1]*h[3]/ϵ(h)) / (h[1] + im*h[2])
function σx_cv(h::GeneralTwoBand) 
    numerator       = :(im*$(hy(h)) + $(hx(h))*$(hz(h)) / $(ϵ(h)))
    denominator     = :($(hx(h)) + im * $(hy(h))) 
    return :($numerator / $denominator)
end

σy_cv(h::GeneralTwoBand,kx,ky)          = σy_cv(hx(h,kx,ky),hy(h,kx,ky),hz(h,kx,ky))
σy_cv(hx::Number,hy::Number,hz::Number) = (-im * hx + hy*hz/ϵ(hx,hy,hz)) / (hx + im*hy)
σy_cv(h::SVector{3,<:Number})           = (-im * h[1] + h[2]*h[3]/ϵ(h)) / (h[1] + im*h[2])
function σy_cv(h::GeneralTwoBand)
    numerator       = :(-im*$(hx(h)) + $(hy(h))*$(hz(h)) / $(ϵ(h)))
    denominator     = :($(hx(h)) + im * $(hy(h))) 
    return :($numerator / $denominator)
end

σz_cv(h::GeneralTwoBand,kx,ky)          = σz_cv(hx(h,kx,ky),hy(h,kx,ky),hz(h,kx,ky))
σz_cv(hx::Number,hy::Number,hz::Number) = (-hx + im*hy) / ϵ(hx,hy,hz)
σz_cv(h::SVector{3,<:Number})           = (-h[1] + im*h[2]) / ϵ(h)
σz_cv(h::GeneralTwoBand)                = :((-$(hx(h)) + im*$(hy(h))) / $(ϵ(h)))

σx_vc(h::GeneralTwoBand,kx,ky)          = conj(σx_cv(h,kx,ky))
σx_vc(hx::Number,hy::Number,hz::Number) = (-im * hy + hx*hz/ϵ(hx,hy,hz)) / (hx - im*hy)
σx_vc(h::SVector{3,<:Number})           = (-im * h[2] + h[1]*h[3]/ϵ(h)) / (h[1] - im*h[2])
function σx_vc(h::GeneralTwoBand)
    numerator       = :(-im*$(hy(h)) + $(hx(h))*$(hz(h)) / $(ϵ(h)))
    denominator     = :($(hx(h)) - im * $(hy(h))) 
    return :($numerator / $denominator)
end

σy_vc(h::GeneralTwoBand,kx,ky)          = conj(σy_cv(h,kx,ky))
σy_vc(hx::Number,hy::Number,hz::Number) = (im * hx + hy*hz/ϵ(hx,hy,hz)) / (hx - im*hy)
σy_vc(h::SVector{3,<:Number})           = (im * h[1] + h[2]*h[3]/ϵ(h)) / (h[1] - im*h[2])
function σy_vc(h::GeneralTwoBand)
    numerator       = :(im*$(hx(h)) + $(hy(h))*$(hz(h)) / $(ϵ(h)))
    denominator     = :($(hx(h)) - im * $(hy(h))) 
    return :($numerator / $denominator)
end

σz_vc(h::GeneralTwoBand,kx,ky)          = conj(σz_cv(h,kx,ky))
σz_vc(hx::Number,hy::Number,hz::Number) = (-hx - im*hy) / ϵ(hx,hy,hz)
σz_vc(h::SVector{3,<:Number})           = (-h[1] - im*h[2]) / ϵ(h)
σz_vc(h::GeneralTwoBand)                = :((-$(hx(h)) - im*$(hy(h))) / $(ϵ(h)))

σvec_cv(h::Union{SVector{3,<:Number},GeneralTwoBand}) = SA[σx_cv(h),σy_cv(h),σz_cv(h)]

σvec_vc(h::Union{SVector{3,<:Number},GeneralTwoBand}) = SA[σx_vc(h),σy_vc(h),σz_vc(h)]
