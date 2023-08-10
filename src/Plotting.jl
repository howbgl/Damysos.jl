function plottimeseries(timeseries::Vector{Vector{T}},
                        labels::Vector{String},
                        tsamples::Vector{Vector{T}};
                        title="",
                        sidelabel="",
                        colors="categorical",
                        kwargs...) where {T<:Real}

    f   = Figure()
    ax  = Axis(f[1,1],title=title,xlabel="t/tc")
    
    for (i,data,label,ts) in zip(1:length(timeseries),timeseries,labels,tsamples)
        
        if colors == "categorical"
            lines!(ax,ts,data,label=label)
        elseif colors == "continuous"
            lines!(ax,ts,data,label=label,colormap=:viridis,
                colorrange=(1,length(timeseries)),color=i)
        else
            @warn "Unknown kwarg colors = $colors, using default (categorical)"
            lines!(ax,ts,data,label=label)
        end
        
    end

    axislegend(ax,position=:lb)
    Label(f[1,2],sidelabel,tellheight=false,justification = :left)
    
    return f
end

function plotspectra(timeseries::Vector{Vector{T}},
                    labels::Vector{String},
                    frequencies::Vector{T},
                    timesteps::Vector{T};
                    maxharm=30,
                    fftwindow=hanning,
                    title="",
                    sidelabel="",
                    colors="categorical",
                    kwargs...) where {T<:Real}

    f   = Figure()
    ax  = Axis(f[1,1],
                title=title,
                xlabel="Ω/ω",
                ylabel="|I|²",
                yscale=log10,
                xminorticksvisible=true,
                xminorgridvisible=true,
                xminorticks=0:1:maxharm,
                xticks=0:5:maxharm)
    xlims!(ax,[0,maxharm])

    for (i,data,label,dt,ν) in zip(1:length(timeseries),timeseries,labels,timesteps,
                                    frequencies)

        pdg         = periodogram(data,
                                    nfft=8*length(data),
                                    fs=1/dt,
                                    window=fftwindow)
        ydata       = pdg.power .* (pdg.freq .^ 2)
        # ydata       = ydata / maximum(ydata)
        xdata       = 1/ν .* pdg.freq
        cut_inds    = ydata .> floatmin(T)
        if length(ydata[cut_inds]) < length(ydata)
            @info "Removing zeros/negatives in plotting spectrum of $title ($label)"
        end

        if colors == "categorical"
            lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label) 
        elseif colors == "continuous"
            lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label,colormap=:viridis,
            colorrange=(1,length(timeseries)),color=i) 
        else
            @warn "Unknown kwarg colors = $colors, using default (categorical)"
            lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label)
        end
        
    end

    axislegend(ax,position=:lb)
    Label(f[1,2],sidelabel,tellheight=false,justification = :left)
    
    return f
end

function plotdata(ens::Ensemble{T};maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}
    
    for obs in ens[1].observables
        plotdata(ens,obs;maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end
end


function plotdata(ens::Ensemble{T},vel::Velocity{T};
        maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}
    
        plotpath    = ens.plotpath

        for (vsymb,vname) in zip([:vx,:vxintra,:vxinter,:vy,:vyintra,:vyinter],
                                ["vx","vxintra","vxinter","vy","vyintra","vyinter"])

            timeseries  = Vector{Vector{T}}(undef,0)
            tsamples    = Vector{Vector{T}}(undef,0)
            timesteps   = Vector{T}(undef,0)
            frequencies = Vector{T}(undef,0)
            labels      = Vector{String}(undef,0)

            for sim in ens.simlist
                pars    = getparams(sim)
                v       = filter(x -> x isa Velocity,sim.observables)[1]
                data    = getproperty(v,vsymb)

                push!(timeseries,data)
                push!(tsamples,pars.tsamples)
                push!(timesteps,pars.dt)
                push!(frequencies,pars.ν)
                push!(labels,sim.id)
            end
            
            try
                figtime     = plottimeseries(timeseries,labels,tsamples,
                                            title=vname,
                                            colors="continuous",
                                            kwargs...)
                figspectra  = plotspectra(timeseries,labels,frequencies,timesteps,
                                            maxharm=maxharm,
                                            fftwindow=fftwindow,
                                            title=vname,
                                            colors="continuous",
                                            kwargs...)
                
                altpath             = joinpath(pwd(),basename(plotpath))
                (success,plotpath)  = ensurepath([plotpath,altpath])
                if success
                    CairoMakie.save(joinpath(plotpath,vname*".pdf"),figtime)
                    CairoMakie.save(joinpath(plotpath,vname*"_spec.pdf"),figspectra)
                    @info "Saved $(vname).pdf & $(vname).spec.pdf at $(plotpath)"
                else
                    @warn "Could not save $(vname) plots."
                end 

            catch e
                @warn "In plotdata(ens::Ensemble{T},vel::Velocity{T};...)",e
            end            
        end        
end


function plotdata(ens::Ensemble{T},occ::Occupation{T};
    maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

    timeseries  = Vector{Vector{T}}(undef,0)
    labels      = Vector{String}(undef,0)
    plotpath    = ens.plotpath

    for sim in ens.simlist
        p       = getparams(sim)
        o       = filter(x -> x isa Occupation,sim.observables)[1]
        data    = o.cbocc

        push!(timeseries,data)
        push!(labels,sim.id)
    end

    try
        figtime     = plottimeseries(timeseries,labels,p.tsamples,
                                    title="CB occupation",
                                    kwargs...)
        figspectra  = plotspectra(timeseries,labels,p.ν,p.dt,
                                    maxharm=maxharm,
                                    fftwindow=fftwindow,
                                    title="CB occupation",
                                    kwargs...)

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"cb_occ.pdf"),figtime)
            CairoMakie.save(joinpath(plotpath,"cb_occ_spec.pdf"),figspectra)
            @info "Saved cb_occ.pdf & cb_occ_spec.pdf at $(plotpath)"
        else
            @warn "Could not save occupation plots."
        end

    catch e
        @warn "In plotdata(ens::Ensemble{T},occ::Occupation{T};...)",e
    end
end

function plotdata(sim::Simulation{T};fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
    
    for obs in sim.observables
        plotdata(sim,obs;fftwindow=fftwindow,maxharm=maxharm,kwargs...)        
    end

    plotfield(sim)

    return nothing
end


function plotdata(sim::Simulation{T},vel::Velocity{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}

    p           = getparams(sim)
    plotpath    = sim.plotpath

    timeseriesx = [vel.vx,vel.vxintra,vel.vxinter]
    tsamplesx   = collect.([p.tsamples,p.tsamples,p.tsamples])
    timestepsx  = [p.dt,p.dt,p.dt]
    frequenciesx = [p.ν,p.ν,p.ν]
    labelsx     = ["vx", "vxintra", "vxinter"]

    timeseries  = [timeseriesx]
    tsamples    = [tsamplesx]
    timesteps   = [timestepsx]
    frequencies = [frequenciesx]
    labels      = [labelsx]

    if sim.dimensions==2
        push!(timeseries,[vel.vy,vel.vyintra,vel.vyinter])
        push!(tsamples,tsamplesx)
        push!(timesteps,timestepsx)
        push!(frequencies,frequenciesx)
        push!(labels,["vy","vyintra","vyinter"])
    end

    try
        for (data,lab,ts,dt,ν) in zip(timeseries,labels,tsamples,timesteps,frequencies)

            figtime     = plottimeseries(data,lab,ts,
                                    title=sim.id,
                                    sidelabel=printparamsSI(sim),
                                    kwargs...)
            figspectra  = plotspectra(data,lab,ν,dt,
                                        maxharm=maxharm,
                                        fftwindow=fftwindow,
                                        title=sim.id,
                                        sidelabel=printparamsSI(sim),
                                        kwargs...)

            altpath             = joinpath(pwd(),basename(plotpath))
            (success,plotpath)  = ensurepath([plotpath,altpath])
            if success
                CairoMakie.save(joinpath(plotpath,"$(lab[1]).pdf"),figtime)
                CairoMakie.save(joinpath(plotpath,"$(lab[1])_spec.pdf"),figspectra)
                @info "Saved $(lab[1]).pdf & $(lab[1]).spec.pdf at $(plotpath)"
            else
                @warn "Could not save $((lab[1])) plots."
            end
        end
        
    catch e
        @warn "In plotdata(sim::Simulation{T},vel::Velocity{T};...)",e
    end

    return nothing
end


function plotdata(sim::Simulation{T},occ::Occupation{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
   
    p           = getparams(sim)
    plotpath    = sim.plotpath
    try
        figtime     = plottimeseries([occ.cbocc],["CB occupation"],[p.tsamples],
                title=sim.id,
                sidelabel=printparamsSI(sim),
                kwargs...)
        figspectra  = plotspectra([occ.cbocc],["CB occupation"],[p.ν],[p.dt],
                maxharm=maxharm,
                fftwindow=fftwindow,
                title=sim.id,
                sidelabel=printparamsSI(sim),
                kwargs...)

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"cb_occ.pdf"),figtime)
            CairoMakie.save(joinpath(plotpath,"cb_occ_spec.pdf"),figspectra)
            @info "Saved cb_occ.pdf & cb_occ_spec.spec.pdf at $(plotpath)"
        else
            @warn "Could not save occupation plots."
        end

    catch e
        @warn "In plotdata(sim::Simulation{T},occ::Occupation{T};...)",e
    end
    
end


function plotfield(sim::Simulation{T}) where {T<:Real}

    name    = getname(sim)
    p       = getparams(sim)
    ts      = collect(p.tsamples)
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)

    plotpath = sim.plotpath
    try
        figa    = plottimeseries([ax.(ts),ay.(ts)],["Ax","Ay"],[ts,ts],
                    title=name,sidelabel=printparamsSI(sim))
        fige    = plottimeseries([ex.(ts),ey.(ts)],["Ex","Ey"],[ts,ts],
                    title=name,sidelabel=printparamsSI(sim))

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"vecfield.pdf"),figa)
            CairoMakie.save(joinpath(plotpath,"efield.pdf"),fige)
            @info "Saved vecfield.pdf & efield.spec.pdf at $(plotpath)"
        else
            @warn "Could not save driving field plots."
        end
    catch e
        @warn "In plotfield(sim::Simulation{T})",e
    end
end

function plotbandstructure(sim::Simulation{T};plotkgrid=true) where {T<:Real}
    
    if sim.dimensions==2
        return plotbandstructure2d(sim;plotkgrid=plotkgrid)
    elseif sim.dimensions==1
        return plotbandstructure1d(sim;plotkgrid=plotkgrid)
    end
end


function plotbandstructure2d(sim::Simulation{T};plotkgrid=true) where {T<:Real}
    
    p       = getparams(sim)
    kxs     = p.kxsamples
    kys     = p.kysamples
end


function plotbandstructure1d(sim::Simulation{T};plotkgrid=true) where {T<:Real}
    
    p       = getparams(sim)
    kxs     = p.kxsamples
end