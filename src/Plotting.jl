
function plotdata(ens::Ensemble{T},obs;
        plotpath="/home/how09898/phd/plots/hhgjl/",maxharm=50,kwargs...) where {T<:Real}

    ensemblename = lowercase(getshortname(ens))*ens.name
    if !isdir(plotpath*ensemblename)
        mkpath(plotpath*ensemblename)
    end

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
                                    window=blackman)
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
                savefig(figs,plotpath*ensemblename*'/'*allobsnames[i]*".pdf")
                savefig(fftfigs,plotpath*ensemblename*'/'*allobsnames[i]*"_spec.pdf")
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
        if !isdir(obspath)
            mkpath(obspath)
        end
        plotdata(obs,alldata,p,obspath;fftwindow=fftwindow,kwargs...)
        
    end

    println("Saved plots at ",sim.plotpath)
    return nothing
end

function plotdata(obs::Observable{T},alldata::DataFrame,p,plotpath::String;
                fftwindow=hanning,maxharm=50,kwargs...) where {T<:Real}

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
