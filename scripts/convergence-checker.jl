

function adjust_density(samples::Vector{<:Number}, desired_samples::Int) 
    n = length(samples)

    # Create an interpolation object for the original samples
    itp = interpolate(samples, BSpline(Cubic))

    # Calculate the interpolation positions for the desired number of samples
    interpolation_positions = range(1, stop = n, length = desired_samples)

    # Interpolate the values at the new positions
    interpolated_values = itp[interpolation_positions]

    return interpolated_values
end

function upsample!(a::Vector{<:Number},b::Vector{<:Number})
    
    la = length(a)
    lb = length(b)
    if la==lb
        return
    elseif la<lb
        buf = adjust_density(a,lb)
        resize!(a,lb)
        a .= buf
        return
    else
        buf = adjust_density(b,la)
        resize!(b,la)
        b .= buf
        return
    end
end

function downsample!(a::Vector{<:Number},b::Vector{<:Number})

    la = length(a)
    lb = length(b)

    if la == lb
        return
    elseif la > lb
        buf = adjust_density(a,lb)
        resize!(a,lb)
        a .= buf
        return
    else
        buf = adjust_density(b,la)
        resize!(b,la)
        b .= buf
        return
    end
end

function converged(
    s1::Simulation,
    s2::Simulation)
    
    p1      = getparams(s1)
    p2      = getparams(s2)
    rtol    = maximum([p1.rtol,p2.rtol])
    atol    = maximum([p1.atol,p2.atol])

    res  = [converged(o1,o2,rtol=rtol,atol=atol) for (o1,o2) in zip(s1.observables,s2.observables)]

    return all(res)
end

function converged(
    v1::Velocity,
    v2::Velocity;
    rtol::Real=1e-10,
    atol::Real=1e-10)

    vx1,vx2,vy1,vy2 = deepcopy.([v1.vx,v2.vx,v1.vy,v2.vy])
    upsample!(vx1,vx2)
    upsample!(vy1,vy2)
    
    return all([
        isapprox(vx1,vx2,atol=atol,rtol=rtol),
        isapprox(vy1,vy2,atol=atol,rtol=rtol)])
end

function converged(
    o1::Occupation,
    o2::Occupation;
    rtol::Real=1e-10,
    atol::Real=1e-10)

    cb1,cb2 = deepcopy.([o1.cbocc,o2.cbocc])
    upsample!(cb1,cb2)
    
    return isapprox(cb1,cb2,atol=atol,rtol=rtol)
end

function findminimum_precision(s1::Simulation,s2::Simulation;max_atol=0.1,max_rtol=0.1)

    !isapprox(s1,s2;atol=max_atol,rtol=max_rtol) && return (Inf,Inf)
    min_achieved_atol = max_atol
    min_achieved_rtol = max_atol

    p1 = getparams(s1)
    p2 = getparams(s2)

    min_possible_atol = maximum([p1.atol,p2.atol])
    min_possible_rtol = maximum([p1.rtol,p2.rtol])

    # Sweep the range of tolerance exponentially (i.e. like 1e-2,1e-3,1e-4,...)
    atols = exp10.(log10(max_atol):-1.0:log10(min_possible_atol))
    rtols = exp10.(log10(max_rtol):-1.0:log10(min_possible_rtol))

    # First find the lowest atol, since that is usually less problematic
    for atol in atols
        if isapprox(s1,s2;atol=atol,rtol=rtols[1])
            min_achieved_atol = atol
        else
            break
        end
    end
    for rtol in rtols
        if isapprox(s1,s2;atol=min_achieved_atol,rtol=rtol)
            min_achieved_rtol = rtol
        else
            break
        end
    end

    return (min_achieved_atol,min_achieved_rtol)
end

function findminimum_precision(
    s1::Simulation,
    s2::Simulation,
    atols::AbstractVector{<:Real},
    rtols::AbstractVector{<:Real})

    !isapprox(s1,s2;atol=atols[1],rtol=rtols[1]) && return (Inf,Inf)

    min_achieved_atol = atols[1]
    min_achieved_rtol = rtols[1]

    # First find the lowest atol, since that is usually less problematic
    for atol in atols
        if isapprox(s1,s2;atol=atol,rtol=rtols[1])
            min_achieved_atol = atol
        else
            break
        end
    end
    for rtol in rtols
        if isapprox(s1,s2;atol=min_achieved_atol,rtol=rtol)
            min_achieved_rtol = rtol
        else
            break
        end
    end

    return (min_achieved_atol,min_achieved_rtol)
end

function findminimum_precision(s1::Simulation,s2::Simulation;max_atol=0.1,max_rtol=0.1)

    p1 = getparams(s1)
    p2 = getparams(s2)

    min_possible_atol = maximum([p1.atol,p2.atol])
    min_possible_rtol = maximum([p1.rtol,p2.rtol])

    # Sweep the range of tolerance exponentially (i.e. like 1e-2,1e-3,1e-4,...)
    atols = exp10.(log10(max_atol):-1.0:log10(min_possible_atol))
    rtols = exp10.(log10(max_rtol):-1.0:log10(min_possible_rtol))

    min_achieved_atol,min_achieved_rtol = findminimum_precision(s1,s2,atols,rtols)
    
    # Search the order of magnitude linearly to get a more precise estimate
    atols = LinRange(min_achieved_atol,0.1min_achieved_atol,10)
    rtols = LinRange(min_achieved_rtol,0.1min_achieved_rtol,10)

    return findminimum_precision(s1,s2,atols,rtols)
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


