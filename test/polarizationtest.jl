using CUDA
using CSV
using Damysos
using DataFrames
using FiniteDifferences
using LoggingExtras
using TerminalLoggers
using Test

refsim = Damysos.load_obj_hdf5("referencedata2d_small.hdf5")

function rotate_velocity(v::Velocity, angle::Real)
    # Rotate the velocity vector by a given angle in radians
    c = cos(angle)
    s = sin(angle)
    r = [c -s; s c]
    vx_rotated = v.vx * cos_angle - v.vy * sin_angle
    vy_rotated = v.vx * sin_angle + v.vy * cos_angle
    return Velocity(vx_rotated, v.vxintra, v.vxinter, vy_rotated, v.vyintra, v.vyinter)    
end

function test_pol(v_ref::Velocity,o_ref::Occupation,sim::Simulation,fns,solver::DamysosSolver;
	atol = 1e-10,
	rtol = 1e-2)
    
    res = run!(sim, fns, solver; saveplots = false, 
		savepath = joinpath("testresults",Damysos.getname(sim)))
	v   = filter(o -> o isa Velocity,res)[1]
    o   = filter(o -> o isa Occupation,res)[1]

	return isapprox(v, v_ref, atol = atol, rtol = rtol) && 
              isapprox(o, o_ref, atol = atol, rtol = rtol)
end

function make_test_simulation_pol(df::DrivingField = GaussianAPulseX(3.0,2π,10.0))

    h       = GappedDirac(0.1)
    l       = TwoBandDephasingLiouvillian(h,Inf,0.5)
    tgrid   = SymmetricTimeGrid(0.01,-5df.σ)
    amax    = Damysos.maximum_vecpot(df)

    # kgrid must be pretty fine because bzmask depends on dk for 2d pol, but not for 
    # GaussianAPulseX, so a few more ky points contribute for GaussianAPulseX
    kgrid   = CartesianKGrid2d(0.01,3amax,0.01,0.5amax)
    grid    = NGrid(kgrid, tgrid)
    obs     = [Velocity(grid), Occupation(grid)]
    us      = scaledriving_frequency(u"1THz",u"4.3e5m/s")
    id      = "reference2dsmall"

    return Simulation(l, df, grid, obs, us, id)
end

