
function adjust_density(samples::Vector{T}, desired_samples::Int) where T
    n = length(samples)

    # Create an interpolation object for the original samples
    itp = interpolate(samples, BSpline(Linear()), OnGrid())

    # Calculate the interpolation positions for the desired number of samples
    interpolation_positions = range(1, stop = n, length = desired_samples)

    # Interpolate the values at the new positions
    interpolated_values = itp[interpolation_positions]

    return interpolated_values
end

function compare_spectra(s1::Simulation{T},s2::Simulaiont{T},
        threshold::T,o::Observable{T}) where T<:Real

    params1     = getparams(s1)
    params2     = getparams(s2)

    dt1         = params1.dt
    dt2         = params2.dt
    ν1          = params1.ν
    ν2          = params2.ν
    
    obs1       = filter(x -> x isa typeof(o))
    
    p1 = periodogram(data1,nfft=8*length(data1),fs=1/dt1)
    p2 = periodogram(data2,nfft=8*length(data1),fs=1/dt2)

    ydata1 = p1.power .* (p1.freq .^ 2)
    ydata2 = p2.power .* (p2.freq .^ 2)
    xdata1 = p1.freq ./ ν1
    xdata2 = p2.freq ./ ν2


    
end

function relative_deviation_L2(s1::Simulation{T},s2::Simulaiont{T}) where T<:Real
    
    deviations  = Vector{T}()
    observables = Vector{Observable{T}}()

    for (o1,o2) in zip(s1.observables,s2.observables)
        push!(observables,o1)
        push!(deviations,relative_deviation_L2(o1,o2))
    end
end

function relative_deviation_L2(v1::Velocity{T},v2::Velocity{T}) where T<:Real
    
end

function relative_deviation_L2_vx(s1::Simulation{T},s2::Simulation{T}) where T<:Real
    
    v1 = filter(x -> x isa Velocity,s1.observables)[1]
    v2 = filter(x -> x isa Velocity,s2.observables)[1]

    return relative_deviation_L2(v1.vx,v2.vx)
end

function relative_deviation_L2(data1::Vector,data2::Vector)
    
    max_samples = maximum(length.([data1,data2]))
    samples     = collect(range(1,max_samples))
    data1       = adjust_density(data1,max_samples)
    data2       = adjust_density(data2,max_samples)

    l2norm      = trapz(samples,abs.(data1) .^2)
    absdev      = trapz(samples,abs.(data1 .- data2) .^2)

    return absdev / l2norm
end

function converge_dt_vx(s::Simulation{T};tol=1e-6) where T<:Real
    
    converged = false
    @info "Running initial sim"
    run_simulation!(s;kxparallel=true)

    currentsim = s

    while converged==false
        dkx      = currentsim.numericalparams.dkx
        dky      = currentsim.numericalparams.dky
        kxmax    = currentsim.numericalparams.kxmax
        kymax    = currentsim.numericalparams.kymax
        t0       = currentsim.numericalparams.t0
        dt       = 0.8 * currentsim.numericalparams.dt

        nextpars = NumericalParams2d(dkx,dky,kxmax,kymax,dt,t0)
        nextsim  = Simulation(
                                currentsim.hamiltonian,
                                currentsim.drivingfield,
                                nextpars,
                                deepcopy(currentsim.observables),
                                currentsim.unitscaling,
                                currentsim.dimensions)

        @info "Running next sim"
        run_simulation!(nextsim,kxparallel=true)

        reldev   = relative_deviation_L2_vx(currentsim,nextsim)
        @show reldev

        if abs(reldev) < tol
            converged = true
        else
            currentsim = nextsim
        end

    end
    @info "Converged!"
end


