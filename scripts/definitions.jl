using Damysos,StaticArrays,Distributed

import Damysos.buildrhs_cc_cv_x_expression
import Damysos.buildobservable_expression_upt
import Damysos.buildbzmask_expression_upt

h       = GappedDirac(0.1)
l       = TwoBandDephasingLiouvillian(h,Inf,Inf)
df      = GaussianAPulse(1.0,2π,0.1)
pars    = NumericalParams2d(0.1,2.0,100,100,0.01,-5)
us      = UnitScaling(1.,1.)
sim     = Simulation(l,df,pars,[Velocity(pars)],us,2)
fns     = define_functions(sim,LinearChunked(128))

rhsccex,rhscvex = buildrhs_cc_cv_x_expression(sim)

@everywhere @eval rhscc(cc,cv,kx,ky,t)  = $rhsccex
@everywhere @eval rhscv(cc,cv,kx,ky,t)  = $rhscvex
@everywhere @eval obs(u,p,t)            = $(buildobservable_expression_upt(sim))
@everywhere @eval bzmask(p,t)           = $(buildbzmask_expression_upt(sim))

@everywhere @eval rhs(u,p,t)        = $(buildrhs_x_expression(sim))
@everywhere @eval bzmask(kx,ky,t)   = $(buildbzmask_expression(sim))
@everywhere @eval f(u,kx,ky,t)      = $(buildobservable_expression(sim))
