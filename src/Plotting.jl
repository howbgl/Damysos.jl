

function plottimeseries(timeseries::Vector{Vector{T}},labels::Vector{String},tsamples;
                        title="",
                        sidelabel="",
                        kwargs...) where {T<:Real}

    f   = Figure()
    ax  = Axis(f[1,1],title=title,xlabel="t/tc")

    for (data,label) in zip(timeseries,labels)
        
        lines!(ax,tsamples,data,label=label)
    end

    axislegend(ax)
    Label(f[1,2],sidelabel,tellheight=false,justification = :left)
    
    return f
end

function plotspectra(timeseries::Vector{Vector{T}},labels::Vector{String},freq,dt;
                    maxharm=30,
                    fftwindow=hanning,
                    title="",
                    sidelabel="",
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

    for (data,label) in zip(timeseries,labels)

        pdg         = periodogram(data,
                                    nfft=8*length(data),
                                    fs=1/dt,
                                    window=fftwindow)
        ydata       = pdg.power .* (pdg.freq .^ 2)
        ydata       = ydata / maximum(ydata)
        xdata       = 1/freq .* pdg.freq
        cut_inds    = ydata .> floatmin(T)
        if length(ydata[cut_inds]) < length(ydata)
            println("Removing zeros/negatives in plotting spectrum of $title ($label)")
        end
        lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label) 
    end

    axislegend(ax)
    Label(f[1,2],sidelabel,tellheight=false,justification = :left)
    
    return f
end

function plotdata(ens::Ensemble{T};maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}
    ensemblename = getname(ens)
    ensurepath(ens.plotpath*ensemblename)

    for obs in ens[1].observables
        plotdata(ens,obs;maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end
end


function plotdata(ens::Ensemble{T},vel::Velocity{T};
        maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

        for (vsymb,vname) in zip([:vx,:vxintra,:vxinter],["vx","vxintra","vxinter"])

            timeseries  = Vector{Vector{T}}(undef,0)
            labels      = Vector{String}(undef,0)

            for sim in ens.simlist
                p       = getparams(sim)
                v       = filter(x -> x isa Velocity,sim.observables)[1]
                data    = getproperty(v,vsymb)

                push!(timeseries,data)
                push!(labels,sim.id)
                plottimeseries(ax,p.tsamples,data,label=sim.id)

                plotspectra(fftax,data,p.ν,p.dt,label=sim.id,title=vname,maxharm=maxharm,
                                fftwindow=fftwindow,kwargs...)
            end
            
            figtime     = plottimeseries(timeseries,labels,p.tsamples,title=vname,kwargs...)
            figspectra  = plotspectra(timeseries,labels,p.ν,p.dt,
                                        maxharm=maxharm,
                                        fftwindow=fftwindow,
                                        title=vname,
                                        kwargs...)

            CairoMakie.save(ens.plotpath*vname*".pdf",figtime)
            CairoMakie.save(ens.plotpath*vname*"_spec.pdf",figspectra)            
        end        
end


function plotdata(ens::Ensemble{T},occ::Occupation{T};
    maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

    timeseries  = Vector{Vector{T}}(undef,0)
    labels      = Vector{String}(undef,0)

    for sim in ens.simlist
        p       = getparams(sim)
        o       = filter(x -> x isa Occupation,sim.observables)[1]
        data    = o.cbocc

        push!(timeseries,data)
        push!(labels,sim.id)
    end

    figtime     = plottimeseries(timeseries,labels,p.tsamples,
                                title="CB occupation",
                                kwargs...)
    figspectra  = plotspectra(timeseries,labels,p.ν,p.dt,
                                maxharm=maxharm,
                                fftwindow=fftwindow,
                                title="CB occupation",
                                kwargs...)

    CairoMakie.save(ens.plotpath*"cb_occ.pdf",figtime)
    CairoMakie.save(ens.plotpath*"cb_occ_spec.pdf",figspectra)
end

function plotdata(sim::Simulation{T};fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
    
    filename    = getname(sim)

    for obs in sim.observables
        ensurepath(sim.plotpath)
        plotdata(sim,obs;fftwindow=fftwindow,maxharm=maxharm,kwargs...)        
    end

    println("Saved plots at ",sim.plotpath)
    return nothing
end


function plotdata(sim::Simulation{T},vel::Velocity{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}

    p           = getparams(sim)
    timeseries  = [vel.vx,vel.vxintra,vel.vxinter]
    labels      = ["vx", "vxintra", "vxinter"]

    figtime     = plottimeseries(timeseries,labels,p.tsamples,
                                title=sim.id,
                                sidelabel=printparamsSI(sim),
                                kwargs...)
    figspectra  = plotspectra(timeseries,labels,p.ν,p.dt,
                                maxharm=maxharm,
                                fftwindow=fftwindow,
                                title=sim.id,
                                sidelabel=printparamsSI(sim),
                                kwargs...)

    CairoMakie.save(sim.plotpath*"vx.pdf",figtime)
    CairoMakie.save(sim.plotpath*"vx_spec.pdf",figspectra)

    return nothing
end


function plotdata(sim::Simulation{T},occ::Occupation{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
   
    p           = getparams(sim)
    figtime     = plottimeseries(timeseries,["CB occupation"],p.tsamples,
                                title=sim.id,
                                sidelabel=printparamsSI(sim),
                                kwargs...)
    figspectra  = plotspectra(timeseries,["CB occupation"],p.ν,p.dt,
                                maxharm=maxharm,
                                fftwindow=fftwindow,
                                title=sim.id,
                                sidelabel=printparamsSI(sim),
                                kwargs...)

    CairoMakie.save(sim.plotpath*"cb_occ.pdf",figtime)
    CairoMakie.save(sim.plotpath*"cb_occ_spec.pdf",figspectra)
end


function plotfield(sim::Simulation{T}) where {T<:Real}

    name    = getname(sim)
    p       = getparams(sim)
    ts      = p.tsamples
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)
    figa    = plottimeseries([ax.(ts),ay.(ts)],["Ax","Ay"],ts,
                title=name,sidelabel=printparamsSI(sim))
    fige    = plottimeseries([ex.(ts),ey.(ts)],["Ex","Ey"],ts,
                title=name,sidelabel=printparamsSI(sim))

    CairoMakie.save(sim.plotpath*"/vecfield.pdf",figa)
    CairoMakie.save(sim.plotpath*"/efield.pdf",fige)
end
