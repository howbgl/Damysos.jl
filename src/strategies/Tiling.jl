DEFAULT_K_CHUNK_SIZE = 8192

export buildensemble_tiled
export create_density_matrix_file!
export save_density_matrix!

function solve_and_integrate_tiles!(sim::Simulation;
    savedata=true,
    saveplots=true,
    ensemblesolver=EnsembleThreads(),
    kchunksize=DEFAULT_K_CHUNK_SIZE,
    kwargs...)

    p                   = getparams(sim)
    ensembles,ktiles    = buildensemble_tiled(sim;kchunksize=kchunksize)
    
    for (ens,kxs,kys) in zip(ensembles,ktiles[1],ktiles[2])
        sols = solve(
            ens,
            ensemblesolver;
            trajectories=length(kxs)*length(kys),
            saveat=p.tsamples,
            abstol=p.atol,
            reltol=p.rtol)
        @show tile
        bz  = getmovingbz(sim,kxs,kys)
    end
    
    # if savedata
    #     savedata(sim)
    # end
    # if saveplots
    #     plotdata(sim)
    # end
end

@inline function getkgrid_index(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    return CartesianIndices((length(kxsamples),length(kysamples)))[i]
end

@inline function getkgrid_point(
    i::Integer,
    kxsamples::AbstractVector{<:Real},
    kysamples::AbstractVector{<:Real})

    idx = getkgrid_index(i,kxsamples,kysamples)

    return SA[kxsamples[idx[1]],kysamples[idx[2]]]
end


function buildensemble_tiled(sim::Simulation;kchunksize=DEFAULT_K_CHUNK_SIZE)

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
    ktiles         = make_kspace_tiling(kxs,kys,kchunksize)

    ensprob = [EnsembleProblem(prob,
        prob_func = (prob,i,repeat) -> remake(prob,p = getkgrid_point(i,kkx,kky)),
        output_func = (sol, i) -> (vector_of_svec_to_matrix(sol.u), false),
        safetycopy=false) for (kkx,kky) in zip(ktiles[1],ktiles[2])]


    return ensprob,ktiles
end

