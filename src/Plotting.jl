function plotdata(ens::Ensemble{T};maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}
    ensemblename = getname(ens)
    ensurepath(ens.plotpath*ensemblename)

    for obs in ens[1].observables
        plotdata(ens,obs;maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end
end


function plotspectrum!(figure,data::Vector{T},freq,dt;maxharm=30,fftwindow=hanning,
                        label="",title="",kwargs...) where {T<:Real}
    pdg         = periodogram(data,nfft=8*length(data),fs=1/dt,window=fftwindow)
    xmax        = minimum([maxharm,maximum(pdg.freq)/freq])
    ydata       = pdg.power .* (pdg.freq .^ 2)
    ydata       = ydata / maximum(ydata)
    xdata       = 1/freq .* pdg.freq
    cut_inds    = ydata .> floatmin(T)
    if length(ydata[cut_inds]) < length(ydata)
        println("Removing zeros/negatives in plotting spectrum of $title ($label)")
    end
    plot!(figure,
        xdata[cut_inds], 
        ydata[cut_inds],
        yscale=:log10,
        xlims=[0,xmax],
        xticks=0:5:xmax,
        xminorticks=0:xmax,
        xminorgrid=true,
        xgridalpha=0.3,
        label=label,
        title=title)

    return nothing
end


function plotdata(ens::Ensemble{T},vel::Velocity{T};
        maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

        for (vsymb,vname) in zip([:vx,:vxintra,:vxinter],["vx","vxintra","vxinter"])

            fig     = plot();
            fftfig  = plot();
            for sim in ens.simlist
                p       = getparams(sim)
                v       = filter(x -> x isa Velocity,sim.observables)[1]
                data    = getproperty(v,vsymb)
                plot!(fig,p.tsamples,data,label=sim.id)

                plotspectrum!(fftfig,data,p.ν,p.dt,label=sim.id,title=vname,maxharm=maxharm,
                                fftwindow=fftwindow,kwargs...)
            end
                savefig(fig,ens.plotpath*vname*".pdf")
                savefig(fftfig,ens.plotpath*vname*"_spec.pdf")            
        end        
end


function plotdata(ens::Ensemble{T},occ::Occupation{T};
    maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

    fig     = plot();
    fftfig  = plot();
    for sim in ens.simlist
        p       = getparams(sim)
        o       = filter(x -> x isa Occupation,sim.observables)[1]
        data    = o.cbocc
        plot!(fig,p.tsamples,data,label=sim.id)

        plotspectrum!(figure,data,p.ν,p.dt,label=sim.id,title="CB occupation",
                        maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end
        savefig(fig,ens.plotpath*"cbocc.pdf")
        savefig(fftfig,ens.plotpath*"cbocc_spec.pdf")
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

    p   = getparams(sim)
    fig = plot(p.tsamples,[vel.vx vel.vxintra vel.vxinter],label=["vx" "vxintra" "vxinter"])

    fftfig = plot();
    periodograms = []
    for (data,name) in zip([vel.vx,vel.vxintra,vel.vxinter],["vx", "vxintra", "vxinter"])
        plotspectrum!(fftfig,data,p.ν,p.dt,label=name,title=name,
                    maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end

    savefig(fig,sim.plotpath*"vx.pdf")
    savefig(fftfig,sim.plotpath*"vx_spec.pdf")

    return nothing
end


function plotdata(sim::Simulation{T},occ::Occupation{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
   
    p       = getparams(sim)
    fig     = plot(p.tsamples,occ.cbocc,label="CB occupation") 
    fftfig  = plot();
    plotspectrum!(fftfig,data,p.ν,p.dt,label="CB",title="CB occupation",
                    maxharm=maxharm,fftwindow=fftwindow,kwargs...)

    savefig(fig,sim.plotpath*"cbocc.pdf")
    savefig(fftfig,sim.plotpath*"cbocc_spec.pdf")
end


function plotfield(sim::Simulation{T}) where {T<:Real}

    name    = getname(sim)
    p       = getparams(sim)
    ts      = p.tsamples
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)
    figa    = plot(ts, [ax.(ts) ay.(ts)],label=["Ax" "Ay"]);
    savefig(figa,sim.plotpath*"/vecfield.pdf")
    fige    = plot(ts, [ex.(ts) ey.(ts)],label=["Ax" "Ay"]);
    savefig(fige,sim.plotpath*"/efield.pdf")

end
