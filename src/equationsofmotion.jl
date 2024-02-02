function get_rhscc_x_expression(l::TwoBandDephasingLiouvillian,df::DrivingField)

    h       = l.hamiltonian
    f       = efieldx(df)
    a       = vecpotx(df)
    Δϵ      = Δϵ(h)
    dcc     = dx_cc(h)
    dcv     = dx_cv(h)
    dvc     = dx_vc(h)
    dvv     = dx_vv(h)

    γ1      = 1 / l.t1
    γ2      = 1 / l.t2

    

end