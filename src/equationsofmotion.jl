
export get_rhs_x
export buildrhs_expression_svec


function buildrhs_expression_svec(l::TwoBandDephasingLiouvillian,df::DrivingField)
    
    rhs_cc,rhs_cv   = buildrhs_cc_cv_expression(l,df)
    rhs             = :(SA[$rhs_cc,$rhs_cv])
    
    replace_expression!(rhs,:cc,:(u[1])) # occupations
    replace_expression!(rhs,:cv,:(u[2])) # coherences
    replace_expression!(rhs,:kx,:(p[1])) # kx momentum
    replace_expression!(rhs,:ky,:(p[2])) # ky momentum
    
    return rhs
end


dipoles_x(h::Hamiltonian) = (dx_cc(h), dx_cv(h), dx_vc(h), dx_vv(h))
dipoles_y(h::Hamiltonian) = (dy_cc(h), dy_cv(h), dy_vc(h), dy_vv(h))

"""
    buildrhs_expression_svec(s::Simulation)

Construct the symbolic right-hand side for equations of motion, returns static vector (SA).
"""
buildrhs_expression_svec(s::Simulation) = buildrhs_expression_svec(s.liouvillian,s.drivingfield)

function buildrhs_cc_cv_expression(l::TwoBandDephasingLiouvillian,df::GaussianAPulseX)

    @info "Specialized EOM for GaussianAPulseX"

    h       = l.hamiltonian
    f       = efieldx(df)
    a       = vecpotx(df)
    Δe      = Δϵ(h)
    
    dcc, dcv, dvc, dvv = dipoles_x(h)

    γ1      = 1 / l.t1
    γ2      = 1 / l.t2

    rhs_cc  = :(2*$f * imag(cv * $dvc) + $γ1 * (1-cc))
    rhs_cv  = :((-$γ2 - im*$Δe)*cv - im*$f *(($dvv - $dcc)*cv + $dcv*(2cc-1) ))

    replace_expression!(rhs_cc,:kx,:(kx-$a))
    replace_expression!(rhs_cv,:kx,:(kx-$a))

    return rhs_cc,rhs_cv
end



function buildrhs_cc_cv_expression(l::TwoBandDephasingLiouvillian,df::DrivingField)
    
    h           = l.hamiltonian
    fx,fy       = efieldx(df), efieldy(df)
    ax,ay       = vecpotx(df), vecpoty(df)
    Δe          = Δϵ(h)

    dxcc, dxcv, dxvc, dxvv = dipoles_x(h)
    dycc, dycv, dyvc, dyvv = dipoles_y(h)

    γ1      = 1 / l.t1
    γ2      = 1 / l.t2

    rhs_cc  = :(2*$fx * imag(cv * $dxvc) + 2*$fy * imag(cv * $dyvc) + $γ1 * (1-cc))
    cv_x    = :(- im*$fx *(($dxvv - $dxcc)*cv + $dxcv*(2cc-1) ))
    cv_y    = :(- im*$fy *(($dyvv - $dycc)*cv + $dycv*(2cc-1) ))
    rhs_cv  = :((-$γ2 - im*$Δe)*cv + $cv_x + $cv_y)

    replace_expression!(rhs_cc,:kx,:(kx-$ax))
    replace_expression!(rhs_cc,:ky,:(ky-$ay))
    replace_expression!(rhs_cv,:kx,:(kx-$ax))
    replace_expression!(rhs_cv,:ky,:(ky-$ay))

    return rhs_cc,rhs_cv
end

"""
    buildrhs_cc_cv_expression(s::Simulation)

Build symbolic right-hand sides for the occupations (cc) and coherences (cv).
"""
function buildrhs_cc_cv_expression(s::Simulation)
    return buildrhs_cc_cv_expression(s.liouvillian,s.drivingfield)
end
