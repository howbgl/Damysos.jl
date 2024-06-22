
export σvec_cv
export σvec_vc
export σx_cv
export σx_cv_i
export σx_cv_r
export σx_vc
export σx_vc_i
export σx_vc_r
export σy_cv
export σy_cv_i
export σy_cv_r
export σy_vc
export σy_vc_i
export σy_vc_r
export σz_cv
export σz_cv_i
export σz_cv_r
export σz_vc
export σz_vc_i
export σz_vc_r

const PAULI_SYMBOLS = (
	:σx_cv, :σx_cv_i, :σx_cv_r, :σx_vc, :σx_vc_i, :σx_vc_r,
	:σy_cv, :σy_cv_i, :σy_cv_r, :σy_vc, :σy_vc_i, :σy_vc_r,
    :σz_cv, :σz_cv_i, :σz_cv_r, :σz_vc, :σz_vc_i, :σz_vc_r)

pone(x::Vararg{<:Real}) = one(eltype(promote(x...)))
pone(x::AbstractArray)  = one(eltype(x))

σx_cv(hx::Real, hy::Real, hz::Real) = (im * hy + hx * hz / ϵ(hx, hy, hz)) / (hx + im * hy)
σx_cv(h::SVector{3, <:Real})            = (im * h[2] + h[1] * h[3] / ϵ(h)) / (h[1] + im * h[2])
function σx_cv(h::GeneralTwoBand)
	numerator   = :(im * $(hy(h)) + $(hx(h)) * $(hz(h)) / $(ϵ(h)))
	denominator = :($(hx(h)) + im * $(hy(h)))
	return :($numerator / $denominator)
end

σx_cv_r(hx::Real, hy::Real, hz::Real) = (hy^2 + hx^2 * hz / ϵ(hx, hy, hz)) / (hx^2 + hy^2)
σx_cv_r(h::SVector{3, <:Real})        = (h[1]^2 + h[1]^2 * h[3] / ϵ(h)) / (h[1]^2 + h[2]^2)
function σx_cv_r(h::GeneralTwoBand)
	numerator   = :(($hy(h))^2 + $(hx(h))^2 * $(hz(h)) / $(ϵ(h)))
	denominator = :(($(hx(h)))^2 + ($(hy(h)))^2)
	return :($numerator / $denominator)
end

σx_cv_i(hx::Real, hy::Real, hz::Real) = hx * hy * (pone(hx, hy, hz) - hz / ϵ(hx, hy, hz)) / (hx^2 + hy^2)
σx_cv_i(h::SVector{3, <:Real})        = h[1] * h[2] * (pone(h) - h[3] / ϵ(h)) / (h[1]^2 + h[2]^2)
function σx_cv_i(h::GeneralTwoBand)
	numerator   = :($(hx(h)) * ($(hy(h)) - $(hy(h)) * $(hz(h)) / $(ϵ(h))))
	denominator = :(($(hx(h)))^2 + ($(hy(h)))^2)
	return :($numerator / $denominator)
end

σy_cv(hx::Real, hy::Real, hz::Real) = (-im * hx + hy * hz / ϵ(hx, hy, hz)) / (hx + im * hy)
σy_cv(h::SVector{3, <:Real})            = (-im * h[1] + h[2] * h[3] / ϵ(h)) / (h[1] + im * h[2])
function σy_cv(h::GeneralTwoBand)
	numerator   = :(-im * $(hx(h)) + $(hy(h)) * $(hz(h)) / $(ϵ(h)))
	denominator = :($(hx(h)) + im * $(hy(h)))
	return :($numerator / $denominator)
end

σy_cv_r(hx::Real, hy::Real, hz::Real) = hx * hy * (hz / ϵ(hx, hy, hz) - pone(hx, hy, hz)) / (hx^2 + hy^2)
σy_cv_r(h::SVector{3, <:Real})            = h[1] * h[2] * (h[3] / ϵ(h) - pone(h)) / (h[1]^2 + h[2]^2)
function σy_cv_r(h::GeneralTwoBand)
	numerator   = :($(h[1]) * ($(h[2]) * $(h[3]) / $(ϵ(h)) - $(h[2])))
	denominator = :(($(h[1]))^2 + ($(h[2]))^2)
	return :($numerator / $denominator)
end

σy_cv_i(hx::Real, hy::Real, hz::Real) = -(hx^2 + hz * hy^2 / ϵ(hx, hy, hz)) / (hx^2 + hy^2)
σy_cv_i(h::SVector{3, <:Real})            = -(h[1]^2 + h[3] * h[2]^2 / ϵ(h)) / (h[1]^2 + h[2]^2)
function σy_cv_i(h::GeneralTwoBand)
	numerator   = :(-$(h[1])^2 - $(h[3]) * ($(h[2]))^2 / $(ϵ(h)))
	denominator = :(($(h[1]))^2 + ($(h[2]))^2)
	return :($numerator / $denominator)
end

σz_cv(hx::Real, hy::Real, hz::Real) = (-hx + im * hy) / ϵ(hx, hy, hz)
σz_cv(h::SVector{3, <:Real})            = (-h[1] + im * h[2]) / ϵ(h)
σz_cv(h::GeneralTwoBand)                  = :((-$(hx(h)) + im * $(hy(h))) / $(ϵ(h)))

σz_cv_r(hx::Real, hy::Real, hz::Real) = -hx / ϵ(hx, hy, hz)
σz_cv_r(h::SVector{3, <:Real})            = -h[1] / ϵ(h)
σz_cv_r(h::GeneralTwoBand)                  = :(-$(h[1]) / $(ϵ(h)))

σz_cv_i(hx::Real, hy::Real, hz::Real) = hy / ϵ(hx, hy, hz)
σz_cv_i(h::SVector{3, <:Real})            = h[2] / ϵ(h)
σz_cv_i(h::GeneralTwoBand)                  = :(-$(h[3]) / $(ϵ(h)))

for func in PAULI_SYMBOLS
    @eval(Damysos,$func(h::GeneralTwoBand,kx::Real,ky::Real) = $func(hx(h, kx, ky), hy(h, kx, ky), hz(h, kx, ky)))
end


σx_vc(hx::Real, hy::Real, hz::Real) = (-im * hy + hx * hz / ϵ(hx, hy, hz)) / (hx - im * hy)
σx_vc(h::SVector{3, <:Real})            = (-im * h[2] + h[1] * h[3] / ϵ(h)) / (h[1] - im * h[2])
function σx_vc(h::GeneralTwoBand)
	numerator   = :(-im * $(hy(h)) + $(hx(h)) * $(hz(h)) / $(ϵ(h)))
	denominator = :($(hx(h)) - im * $(hy(h)))
	return :($numerator / $denominator)
end

σy_vc(hx::Real, hy::Real, hz::Real) = (im * hx + hy * hz / ϵ(hx, hy, hz)) / (hx - im * hy)
σy_vc(h::SVector{3, <:Real})            = (im * h[1] + h[2] * h[3] / ϵ(h)) / (h[1] - im * h[2])
function σy_vc(h::GeneralTwoBand)
	numerator   = :(im * $(hx(h)) + $(hy(h)) * $(hz(h)) / $(ϵ(h)))
	denominator = :($(hx(h)) - im * $(hy(h)))
	return :($numerator / $denominator)
end

σz_vc(hx::Real, hy::Real, hz::Real) = (-hx - im * hy) / ϵ(hx, hy, hz)
σz_vc(h::SVector{3, <:Real})        = (-h[1] - im * h[2]) / ϵ(h)
σz_vc(h::GeneralTwoBand)            = :((-$(hx(h)) - im * $(hy(h))) / $(ϵ(h)))


for (f,g) in zip((:σx_vc_r,:σy_vc_r,:σz_vc_r),(:σx_cv_r,:σy_cv_r,:σz_cv_r))
	@eval(Damysos,$f(hx::Real,hy::Real,hz::Real) = $g(hx::Real,hy::Real,hz::Real))
	@eval(Damysos,$f(h::SVector{3, <:Real}) 	 = $g(h::SVector{3, <:Real}))
	@eval(Damysos,$f(h::GeneralTwoBand)  		 = $g(h::GeneralTwoBand))
end

for (f,g) in zip((:σx_vc_i,:σy_vc_i,:σz_vc_i),(:σx_cv_i,:σy_cv_i,:σz_cv_i))
	@eval(Damysos,$f(hx::Real,hy::Real,hz::Real) = - $g(hx,hy,hz))
	@eval(Damysos,$f(h::SVector{3,<:Real})       = - $g(h))
	@eval(Damysos,$f(h::GeneralTwoBand)          = :( - $(g(h)) ))
end

σvec_cv(h::Union{SVector{3, <:Real}, GeneralTwoBand}) = SA[σx_cv(h), σy_cv(h), σz_cv(h)]

σvec_vc(h::Union{SVector{3, <:Real}, GeneralTwoBand}) = SA[σx_vc(h), σy_vc(h), σz_vc(h)]
