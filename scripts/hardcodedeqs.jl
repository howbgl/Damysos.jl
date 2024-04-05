const Δ = 0.1

Δϵ(kx,ky)   = 2sqrt(kx^2+ky^2+Δ^2)
dxcc(kx,ky) =  ky * (1-Δ/sqrt(kx^2+ky^2+Δ^2)) / 2(kx^2 + ky^2)
dxcv(kx,ky) = (ky/sqrt(kx^2+ky^2+Δ^2) - im*kx*Δ / (kx^2+ky^2+Δ^2)) / 2(kx+im*ky)
dxvc(kx,ky) = conj(dxcv(kx,ky))
dxvv(kx,ky) = -dxcc(kx,ky)