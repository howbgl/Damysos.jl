
function plotdata(ens::Ensemble{T},obs;
        maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

    ensemblename = getname(ens)
    ensurepath(ens.plotpath*ensemblename)

    if length(eachindex(ens.simlist)) == length(eachindex(obs))

        allobsnames = getnames_obs(ens[1])
        arekres     = arekresolved(ens[1])
        for i in eachindex(allobsnames)
            if arekres[i] == true
                continue
            else
                figs    = plot();
                fftfigs = plot();
                for j in eachindex(obs)
                    p           = getparams(ens[j])
                    plot!(figs,p.tsamples,obs[j][i],label=ens[j].id)

                    pdg         = periodogram(obs[j][i],
                                    nfft=8*length(obs[j][i]),
                                    fs=1/p.dt,
                                    window=fftwindow)
                    xmax        = minimum([maxharm,maximum(pdg.freq)/p.ν])
                    ydata       = pdg.power
                    xdata       = 1/p.ν .* pdg.freq
                    cut_inds    = ydata .> floatmin(T)
                    if length(ydata[cut_inds]) < length(ydata)
                        println("Warning: Discarding negative or zero values in plotting\
                         of spectrum of ",allobsnames[i])
                    end
                    plot!(fftfigs,
                        xdata[cut_inds], 
                        ydata[cut_inds],
                        yscale=:log10,
                        xlims=[0,xmax],
                        xticks=0:5:xmax,
                        xminorticks=0:xmax,
                        xminorgrid=true,
                        xgridalpha=0.3,
                        label=ens[j].id)
                end
                savefig(figs,ens.plotpath*allobsnames[i]*".pdf")
                savefig(fftfigs,ens.plotpath*allobsnames[i]*"_spec.pdf")
            end
        end
        
    else
        println("length(eachindex(ens.simlist)) != length(eachindex(obs))\
         in savedata(ens::Ensemble{T},obs,datapath=...)")
        println("Aborting...")
    end
end

function plotdata(sim::Simulation{T};fftwindow=hanning,kwargs...) where {T<:Real}
    p           = getparams(sim)
    filename    = getname(sim)
    alldata     = DataFrame(CSV.File(sim.datapath*filename*"/data.csv"))

    println("Skip plotting k-resolved data for now...")

    for obs in sim.observables
        obspath = sim.plotpath*filename*'/'
        ensurepath(obspath)
        plotdata(obs,alldata,p,obspath;fftwindow=fftwindow,kwargs...)
        
    end

    println("Saved plots at ",sim.plotpath)
    return nothing
end

function plotdata(obs::Observable{T},alldata::DataFrame,p,plotpath::String;
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}

    nonkresolved_obs = []
    periodograms     = []

    for i in eachindex(getnames_obs(obs))
        arekres     = arekresolved(obs)
        obsnames    = getnames_obs(obs)
        if arekres[i] == true
            continue
        else
            push!(nonkresolved_obs,obsnames[i])
            data = getproperty(alldata,Symbol(obsnames[i]))
            push!(periodograms,periodogram(data,
                                nfft=8*length(data),
                                fs=1/p.dt,
                                window=fftwindow))
        end
    end
    nonkresolved_data = select(alldata,nonkresolved_obs)
    xmax    = minimum([maxharm,maximum(periodograms[1].freq)/p.ν])
    fftfig  = plot(1/p.ν .* periodograms[1].freq, 
                hcat([x.power for x in periodograms]...),
                yscale=:log10,
                xlims=[0,xmax],
                xticks=0:5:xmax,
                xminorticks=0:xmax,
                xminorgrid=true,
                xgridalpha=0.3,
                label=permutedims(nonkresolved_obs))
    fig     = plot(p.tsamples,Matrix(nonkresolved_data),label=permutedims(nonkresolved_obs))

    savefig(fig,plotpath*getshortname(obs)*".pdf")
    savefig(fftfig,plotpath*getshortname(obs)*"_spec.pdf")

    return nothing
end

function plotfield(sim::Simulation{T}) where {T<:Real}

    name    = getshortname(sim)
    p       = getparams(sim)
    ts      = p.tsamples
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)
    figa    = plot(ts, [ax.(ts) ay.(ts)],label=["Ax" "Ay"]);
    savefig(figa,sim.plotpath*name*"_vecfield.pdf")
    fige    = plot(ts, [ex.(ts) ey.(ts)],label=["Ax" "Ay"]);
    savefig(fige,sim.plotpath*name*"_efield.pdf")

end
