DEFAULT_K_CHUNK_SIZE = 8192

export buildensemble_linear

function solve_and_integrate_linear!(sim::Simulation;
    savedata=true,
    saveplots=true,
    ensemblesolver=EnsembleThreads(),
    kchunksize=DEFAULT_K_CHUNK_SIZE,
    kwargs...)

    p                   = getparams(sim)
    ensembles,kindices  = buildensemble_linear(sim;kchunksize=kchunksize)
    
    for (ens,inds) in zip(ensembles,kindices)
        sols = solve(
            ens,
            ensemblesolver;
            trajectories=length(kxs)*length(kys),
            saveat=p.tsamples,
            abstol=p.atol,
            reltol=p.rtol)
        @show tile
        bz  = getmovingbz(sim,kxs[inds],kys[inds])
    end
    
    # if savedata
    #     savedata(sim)
    # end
    # if saveplots
    #     plotdata(sim)
    # end
end


function buildensemble_linear(sim::Simulation;kchunksize=DEFAULT_K_CHUNK_SIZE)

    p                   = getparams(sim)
    kxs                 = p.kxsamples
    kys                 = p.kysamples
    γ1                  = one(p.t1) / p.t1
    γ2                  = one(p.t1) / p.t2    
    a                   = get_vecpotx(sim.drivingfield)
    f                   = get_efieldx(sim.drivingfield)
    ϵ                   = getϵ(sim.hamiltonian)
    dcc,dcv,dvc,dvv     = getdipoles_x(sim.hamiltonian)

    rhs_cc(t,cc,cv,kx,ky)  = 2.0 * f(t) * imag(cv * dvc(kx-a(t), ky)) + γ1*(one(t)-cc)
    rhs_cv(t,cc,cv,kx,ky)  = (-γ2 - 2.0im * ϵ(kx-a(t),ky)) * cv - im * f(t) * 
        ((dvv(kx-a(t),ky)-dcc(kx-a(t),ky)) * cv + dcv(kx-a(t),ky) * (2cc - one(t)))
    
    @inline function rhs(u,p,t)
        return SA[rhs_cc(t,u[1],u[2],p[1],p[2]),rhs_cv(t,u[1],u[2],p[1],p[2])]
    end

    u0             = SA[zero(im*p.t1),zero(im*p.t1)]
    tspan          = (p.tsamples[1],p.tsamples[end])
    prob           = ODEProblem{false}(rhs,u0,tspan,getkgrid_point(1,kxs,kys))
    indices        = makeindices_kspace_linearsubdiv(kxs,kys,kchunksize)

    ensprob = [EnsembleProblem(
        prob,
        prob_func   = (prob,i,repeat) -> remake(prob,p = getkgrid_point(inds[i],kxs,kys)),
        output_func = (sol, i) -> (vector_of_svec_to_matrix(sol.u), false),
        safetycopy  = false) for inds in indices]

    return ensprob,indices
end

