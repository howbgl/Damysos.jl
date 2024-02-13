using Damysos,StaticArrays

h       = GappedDirac(0.1)
l       = TwoBandDephasingLiouvillian(h,Inf,Inf)
df      = GaussianAPulse(1.0,2π,0.1)
pars    = NumericalParams2d(1.0,2.0,100,10,0.01,-5)
us      = UnitScaling(1.,1.)
sim     = Simulation(l,df,pars,[Occupation(pars)],us,2)
