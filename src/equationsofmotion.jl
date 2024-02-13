
export buildrhsx
export get_rhs_x
export rhs_x_expression

function buildrhsx(sim::Simulation)

    expr = rhs_x_expression(sim.liouvillian,sim.drivingfield)

    replace_expression!(expr,:cc,:(u[1]))
    replace_expression!(expr,:cv,:(u[2]))
    replace_expression!(expr,:kx,:(p[1]))
    replace_expression!(expr,:ky,:(p[2]))
    
    return @eval (u,p,t) -> $expr
end

function rhs_x_expression(l::TwoBandDephasingLiouvillian,df::DrivingField)

    h       = l.hamiltonian
    f       = efieldx(df)
    a       = vecpotx(df)
    Δe      = Δϵ(h)
    dcc     = dx_cc(h)
    dcv     = dx_cv(h)
    dvc     = dx_vc(h)
    dvv     = dx_vv(h)

    γ1      = 1 / l.t1
    γ2      = 1 / l.t2

    rhs_cc = :(2*$f * imag(cv * $dvc) + $γ1 * (1-cc))
    rhs_cv = :((-$γ2 - im*$Δe)*cv - im*$f *(($dvv - $dcc)*cv + $dcv*(2cc-1) ))

    replace_expression!(rhs_cc,:kx,:(kx-$a))
    replace_expression!(rhs_cv,:kx,:(kx-$a))

    return :(SA[$rhs_cc,$rhs_cv])
end

function get_rhs_x(h::GappedDiracOld,df::DrivingField)

    γ1      = 1/h.t1
    γ2      = 1/h.t2
    a       = get_vecpotx(df)
    f       = get_efieldx(df)
    ϵ       = getϵ(h)
    dcv     = getdx_cv(h)
    dvc     = getdx_vc(h)
    dcc     = getdx_cc(h)
    dvv     = getdx_vv(h)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(1-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2im * ϵ(kx-a(t),ky)) * cv - im * f(t) * 
                        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2cc - 1))

    return rhs_cc,rhs_cv
end
