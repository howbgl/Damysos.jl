DEFAULT_K_CHUNK_SIZE = 512


function solve_and_write!(sim::Simulation;
    savedata=true,
    saveplots=true,
    ensemblesolver=EnsembleThreads(),
    kchunksize=DEFAULT_K_CHUNK_SIZE,
    kwargs...)

    p                   = getparams(sim)
    ensembles,kchunks   = buildensemble(sim;kchunksize=kchunksize)

    filepath = create_density_matrix_file!(sim)
    
    for (ens,c) in zip(ensembles,kchunks)
        sols = solve(
            ens,
            ensemblesolver;
            trajectories=length(c),
            saveat=p.tsamples,
            abstol=p.atol,
            reltol=p.rtol)

        save_density_matrix!(filepath,sols,c,p.kxsamples,p.kysamples)
    end
    
    # if savedata
    #     savedata(sim)
    # end
    # if saveplots
    #     plotdata(sim)
    # end
end

function create_density_matrix_file!(sim::Simulation)

    if ensurepath(sim.temppath)
        filepath    = joinpath(sim.temppath,"densitymatrix.hdf5")
        fid         = h5open(filepath,"w")
        p           = getparams(sim)
        dens_group  = create_group(fid,"densitymatrix")

        create_dataset(
            dens_group,
            "cc",
            datatype(typeof(p.dt)),
            dataspace(p.nkx,p.nky,p.nt))

        create_dataset(
            dens_group,
            "cv",
            datatype(Complex{typeof(p.dt)}),
            dataspace(p.nkx,p.nky,p.nt))
        
        close(fid)
        return filepath
    else
        throw(ErrorException("FATAL: Could not create tempfile for density matrix."))
    end
end

function save_density_matrix!(filepath::String,sols,kindices,kxsamples,kysamples)
    
    fid         = h5open(filepath,"cw")
    dens_group  = fid["densitymatrix"]

    for (sol,ik) in zip(sols,kindices)
        I = getkgrid_index(ik,kxsamples,kysamples)

        dens_group["cc"][I[1],I[2],:] = real.(sol[1,:])
        dens_group["cv"][I[1],I[2],:] = sol[2,:]
    end

    close(fid)

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

function myens(prob,kxs,kys,ts,ex=ThreadedEx())

    idx = CartesianIndices((length(kxs),length(kys)))
    return Folds.collect(
        (solve(remake(prob,p=[kxs[i[1]],kys[i[2]]]);
        saveat=ts,abstol=1e-12,reltol=1e-12) for i in idx),ex)
end


function buildensemble(sim::Simulation;kchunksize=DEFAULT_K_CHUNK_SIZE)

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

    chunks          = subdivide_vector(1:p.nkx*p.nky,kchunksize)
    offsets         = prepend!([last(c) for c in chunks[1:end-1]],0)

    ensprob = [EnsembleProblem(prob,
        prob_func = (prob,i,repeat) -> begin
            k = getkgrid_point(i+d,kxs,kys);
            @debug (i+d,k);
            remake(prob,p = k)
        end,
        output_func = (sol, i) -> (vector_of_svec_to_matrix(sol.u), false),
        safetycopy=false) for d in offsets]


    return ensprob,chunks
end

function prob_func_remake(prob,i,repeat)
    p = getkgrid_point
end

function buildensemble_inplace(sim::Simulation)

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
    
    @inline function rhs!(du,u,p,t)
        du[1] = rhs_cc(t,u[1],u[2],p[1],p[2])
        du[2] = rhs_cv(t,u[1],u[2],p[1],p[2])
    end

    u0             = zeros(Complex{typeof(p.t1)},2)
    tspan          = (p.tsamples[1],p.tsamples[end])
    prob           = ODEProblem{true}(rhs!,u0,tspan,getkgrid_point(1,kxs,kys))
    ensprob        = EnsembleProblem(prob,
        prob_func = (prob,i,repeat) -> remake(prob,p = getkgrid_point(i,kxs,kys)))

    return ensprob
end

