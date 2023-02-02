using DSP, Plots

function plotspectrum(data,frequency,dt;window=rect,kwargs...)
    pdg     = periodogram(data,nfft=8*length(data),fs=1/dt,window=window)
    maxharm = maximum(pdg.freq)/frequency
    return plot(1/frequency .* pdg.freq, pdg.power; 
                    yscale=:log10,
                    xticks=0:5:maxharm,
                    xminorticks=0:maxharm,
                    xminorgrid=true,
                    xgridalpha=0.3,
                    kwargs...)
end
