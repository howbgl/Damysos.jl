
export plotbandstructure
export plotdata
export plotfield


const DEFAULT_FIGSIZE           = (1200,800)
const DEFAULT_COLORSCHEME_CONT  = ColorSchemes.viridis

function plottimeseries(timeseries::Vector{Vector{T}},
                        labels::Vector{String},
                        tsamples::Vector{Vector{T}};
                        title="",
                        sidelabel="",
                        colors="categorical",
                        xlabel="t/tc",
                        ylabel="",
                        kwargs...) where {T<:Real}

    f   = Figure(size=DEFAULT_FIGSIZE)
    ax  = Axis(f[1,1],title=title,xlabel=xlabel,ylabel=ylabel)
    
    for (i,data,label,ts) in zip(1:length(timeseries),timeseries,labels,tsamples)
        
        if colors == "categorical"
            lines!(ax,ts,data,label=label)
        elseif colors == "continuous"
            cs = DEFAULT_COLORSCHEME_CONT
            lines!(ax,ts,data,label=label,color=cs[(i-1)/(length(timeseries)-1)])
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

    f   = Figure(size=DEFAULT_FIGSIZE)
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
            @debug "Removing zeros/negatives in plotting spectrum of $title ($label)"
        end

        if colors == "categorical"
            lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label) 
        elseif colors == "continuous"
            cs = DEFAULT_COLORSCHEME_CONT
            lines!(ax,xdata[cut_inds],ydata[cut_inds],label=label,
                color=cs[(i-1)/(length(timeseries)-1)]) 
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
        @info "Plotting " * getshortname(obs)
        plotdata(ens,obs;maxharm=maxharm,fftwindow=fftwindow,kwargs...)
    end
end


function plotdata(
    ens::Ensemble{T},
    vel::Velocity{T};
    maxharm=30,
    fftwindow=hanning,
    kwargs...) where {T<:Real}
    
    plotpath    = ens.plotpath

    for (vsymb,vname) in zip([:vx,:vxintra,:vxinter,:vy,:vyintra,:vyinter],
                            ["vx","vxintra","vxinter","vy","vyintra","vyinter"])

        timeseries  = Vector{Vector{T}}(undef,0)
        tsamples    = Vector{Vector{T}}(undef,0)
        timesteps   = Vector{T}(undef,0)
        frequencies = Vector{T}(undef,0)
        labels      = Vector{String}(undef,0)

        for sim in ens.simlist
            pars        = getparams(sim)
            lc_in_nm    = ustrip(u"nm",pars.lengthscale)
            d           = sim.dimensions
            ts_in_cyc   = collect(pars.tsamples) .* pars.ν
            v           = filter(x -> x isa Velocity,sim.observables)[1]
            data        = getproperty(v,vsymb) .* pars.vF ./ lc_in_nm^d

            push!(timeseries,data)
            push!(tsamples,ts_in_cyc)
            push!(timesteps,pars.dt)
            push!(frequencies,pars.ν)
            push!(labels,sim.id)
        end
        
        try
            figtime     = plottimeseries(timeseries,labels,tsamples,
                                        title=vname * " (" * ens.id * ")",
                                        colors="continuous",
                                        ylabel=ens[1].dimensions == 1 ? "v [vF nm^-1]" : "v [vF nm^-2]",
                                        kwargs...)
            figspectra  = plotspectra(timeseries,labels,frequencies,timesteps,
                                        maxharm=maxharm,
                                        fftwindow=fftwindow,
                                        title=vname * " (" * ens.id * ")",
                                        colors="continuous",
                                        kwargs...)
            
            altpath             = joinpath(pwd(),basename(plotpath))
            (success,plotpath)  = ensurepath([plotpath,altpath])
            if success
                CairoMakie.save(joinpath(plotpath,vname*".pdf"),figtime)
                CairoMakie.save(joinpath(plotpath,vname*".png"),figtime,px_per_unit = 4)
                CairoMakie.save(joinpath(plotpath,vname*"_spec.pdf"),figspectra)
                CairoMakie.save(joinpath(plotpath,vname*"_spec.png"),figspectra,
                    px_per_unit = 4)
                @debug "Saved $(vname).pdf & $(vname).spec.pdf at\n\"$plotpath\""
            else
                @warn "Could not save $(vname) plots"
            end 

        catch e
            @warn "In plotdata(ens::Ensemble{T},vel::Velocity{T};...)"
            @error e
        end            
    end        
end


function plotdata(ens::Ensemble{T},occ::Occupation{T};
    maxharm=30,fftwindow=hanning,kwargs...) where {T<:Real}

    timeseries  = Vector{Vector{T}}(undef,0)
    tsamples    = Vector{Vector{T}}(undef,0)
    timesteps   = Vector{T}(undef,0)
    frequencies = Vector{T}(undef,0)
    labels      = Vector{String}(undef,0)
    plotpath    = ens.plotpath

    for sim in ens.simlist

        lc      = sim.unitscaling.lengthscale
        pars    = getparams(sim)
        o       = filter(x -> x isa Occupation,sim.observables)[1]
        data    = o.cbocc  / (lc^sim.dimensions)

        push!(timeseries,data)
        push!(tsamples,pars.tsamples)
        push!(timesteps,pars.dt)
        push!(frequencies,pars.ν)
        push!(labels,sim.id)
    end

    try
        figtime     = plottimeseries(timeseries,labels,tsamples,
                                    title="CB occupation" * "(" * ens.id * ")",
                                    colors="continuous",
                                    kwargs...)

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"cb_occ.pdf"),figtime)
            CairoMakie.save(joinpath(plotpath,"cb_occ.png"),figtime,px_per_unit = 4)
            @debug "Saved cb_occ.pdf & cb_occ_spec.pdf at \n\"$plotpath\""
        else
            @warn "Could not save occupation plots."
        end

    catch e
        @warn "In plotdata(ens::Ensemble{T},occ::Occupation{T};...)"
        @error e
    end
end

function plotdata(sim::Simulation{T};fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}
    
    @info "Generating plots"

    for obs in sim.observables
        @info "Plotting " * getshortname(obs)
        plotdata(sim,obs;fftwindow=fftwindow,maxharm=maxharm,kwargs...)        
    end

    plotfield(sim)
    plotbandstructure(sim)
end


function plotdata(sim::Simulation{T},vel::Velocity{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}

    p           = getparams(sim)
    lc_in_nm    = ustrip(u"nm",p.lengthscale)
    d           = sim.dimensions
    ts_in_cyc   = collect(p.tsamples) .* p.ν
    plotpath    = sim.plotpath
    timeseriesx = [p.vF * x ./ lc_in_nm^d for x in [vel.vx,vel.vxintra,vel.vxinter]]
    tsamplesx   = [ts_in_cyc,ts_in_cyc,ts_in_cyc]
    timestepsx  = [p.dt,p.dt,p.dt]
    frequenciesx = [p.ν,p.ν,p.ν]
    labelsx     = ["vx", "vxintra", "vxinter"]

    timeseries  = [timeseriesx]
    tsamples    = [tsamplesx]
    timesteps   = [timestepsx]
    frequencies = [frequenciesx]
    labels      = [labelsx]

    if sim.dimensions==2
        push!(timeseries,[x ./ lc_in_nm^d for x in [vel.vy,vel.vyintra,vel.vyinter]])
        push!(tsamples,tsamplesx)
        push!(timesteps,timestepsx)
        push!(frequencies,frequenciesx)
        push!(labels,["vy","vyintra","vyinter"])
    end

    try
        for (data,lab,ts,dt,ν) in zip(timeseries,labels,tsamples,timesteps,frequencies)

            figtime     = plottimeseries(
                data,lab,ts,
                title=sim.id,
                sidelabel=printparamsSI(sim),
                ylabel=sim.dimensions == 1 ? "v [vF nm^-1]" : "v [vF nm^-2]",
                kwargs...)
            figspectra  = plotspectra(
                data,lab,ν,dt,
                maxharm=maxharm,
                fftwindow=fftwindow,
                title=sim.id,
                sidelabel=printparamsSI(sim),
                kwargs...)

            altpath             = joinpath(pwd(),basename(plotpath))
            (success,plotpath)  = ensurepath([plotpath,altpath])
            if success
                CairoMakie.save(joinpath(plotpath,"$(lab[1]).pdf"),figtime)
                CairoMakie.save(joinpath(plotpath,"$(lab[1]).png"),figtime,px_per_unit = 4)
                CairoMakie.save(joinpath(plotpath,"$(lab[1])_spec.pdf"),figspectra)
                CairoMakie.save(joinpath(plotpath,"$(lab[1])_spec.png"),
                    figspectra,px_per_unit = 4)
                @debug "Saved $(lab[1]).pdf & $(lab[1]).spec.pdf at \n\"$plotpath\""
            else
                @warn "Could not save $((lab[1])) plots"
            end
        end
        
    catch e
        @warn "In plotdata(sim::Simulation{T},vel::Velocity{T};...)"
        @error e
    end

    return nothing
end


function plotdata(sim::Simulation{T},occ::Occupation{T};
                fftwindow=hanning,maxharm=30,kwargs...) where {T<:Real}

    p           = getparams(sim)
    plotpath    = sim.plotpath
    lc_in_nm    = ustrip(u"nm",p.lengthscale)
    d           = sim.dimensions
    data        = occ.cbocc ./ lc_in_nm^d
    ts_in_cyc   = collect(p.tsamples) .* p.ν
    try
        
        figtime     = plottimeseries([data],["CB occupation"],[ts_in_cyc],
                title=sim.id,
                sidelabel=printparamsSI(sim),
                ylabel = sim.dimensions == 1 ? "ρcc [nm^-1]" : "ρcc [nm^-2]",
                kwargs...)

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"cb_occ.pdf"),figtime)
            CairoMakie.save(joinpath(plotpath,"cb_occ.png"),figtime,px_per_unit = 4)
            @debug "Saved cb_occ.pdf & cb_occ_spec.spec.pdf at \"$plotpath\"\n"
        else
            @warn "Could not save occupation plots"
        end

    catch e
        @warn "In plotdata(sim::Simulation{T},occ::Occupation{T};...)"
        @error e
    end
    
end


function plotfield(sim::Simulation{T}) where {T<:Real}

    @info "Plotting driving field"

    name    = getname(sim)
    p       = getparams(sim)
    ts      = collect(p.tsamples)
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)

    ts_in_cyc           = collect(p.tsamples) .* p.ν
    e                   = uconvert(u"C",u"1eV" / u"1V")
    vecpot_SI_factor    = ustrip(u"MV*fs/cm",Unitful.ħ/(e*p.lengthscale))
    field_SI_factor     = ustrip(u"MV/cm",Unitful.ħ/(e*p.lengthscale*p.timescale))

    plotpath = sim.plotpath
    try
        figa    = plottimeseries(
            [ax.(ts) .* vecpot_SI_factor,ay.(ts) .* vecpot_SI_factor],
            ["Ax","Ay"],
            [ts_in_cyc,ts_in_cyc],
            title=name,
            sidelabel=printparamsSI(sim),
            xlabel="time [1/ν]",
            ylabel="vector potential [fs MV/cm]")
        fige    = plottimeseries(
            [ex.(ts) .* field_SI_factor,ey.(ts) .* field_SI_factor],
            ["Ex","Ey"],
            [ts_in_cyc,ts_in_cyc],
            title=name,
            sidelabel=printparamsSI(sim),
            xlabel="time [1/ν]",
            ylabel="el. field [MV/cm]")

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"vecfield.pdf"),figa)
            CairoMakie.save(joinpath(plotpath,"vecfield.png"),figa,px_per_unit = 4)
            CairoMakie.save(joinpath(plotpath,"efield.pdf"),fige)
            CairoMakie.save(joinpath(plotpath,"efield.png"),fige,px_per_unit = 4)
            @debug "Saved vecfield.pdf & efield.spec.pdf at \"$plotpath\"\n"
        else
            @warn "Could not save driving field plots"
        end
    catch e
        @warn "In plotfield(sim::Simulation{T})"
        @error e
    end
end

function plotbandstructure(sim::Simulation;plotkgrid=false)
    
    if sim.dimensions==2
        return plotbandstructure2d(sim;plotkgrid=plotkgrid)
    elseif sim.dimensions==1
        return plotbandstructure1d(sim;plotkgrid=plotkgrid)
    end
end


function plotbandstructure2d(sim::Simulation;plotkgrid=false,nk=2048)
    
    @info "Plotting bandstructure"

    plotpath    = sim.plotpath
    p           = getparams(sim)
    bzSI        = [ustrip(u"Å^-1",wavenumberSI(k,sim.unitscaling)) for k in p.bz]
    bzSI_kx     = [bzSI[1],bzSI[2],bzSI[2],bzSI[1],bzSI[1]] 
    bzSI_ky     = [bzSI[3],bzSI[3],bzSI[4],bzSI[4],bzSI[3]] 
    Δϵ          = getΔϵ(sim.hamiltonian)
    kmax        = maximum([p.kxmax,p.kymax])
    dk          = 2kmax/nk
    ks          = -kmax:dk:kmax
    ksSI        = [ustrip(u"Å^-1",wavenumberSI(k,sim.unitscaling)) for k in ks]
    zdata       = [Δϵ(kx,ky) for kx in ks, ky in ks]
    zdataSI     = [ustrip(u"meV",energySI(en,sim.unitscaling)) for en in zdata]
    fig         = Figure(size=DEFAULT_FIGSIZE .+ (200,0))
    ax          = Axis(fig[1, 1],
        title=sim.id,
        xlabel="kx [Å^-1]",
        ylabel="ky [Å^-1]",
        aspect=1)
    
    try
        cont = contourf!(ax,ksSI,ksSI,zdataSI)
        Colorbar(fig[1,2],cont)
        if plotkgrid
            for ky in p.kysamples
                scatter!(ax,collect(p.kxsamples),fill(ky,p.nkx);color=:black,markersize=1.2)
            end
        end
        lines!(ax,bzSI_kx,bzSI_ky,color=:black)
        tooltip!(bzSI[2],bzSI[4],"Brillouin Zone",offset=0,align=0.8)
        Label(fig[1,3],printparamsSI(sim),tellheight=false,justification = :left)

        altpath             = joinpath(pwd(),basename(plotpath))
        (success,plotpath)  = ensurepath([plotpath,altpath])
        if success
            CairoMakie.save(joinpath(plotpath,"bandstructure.pdf"),fig)
            CairoMakie.save(joinpath(plotpath,"bandstructure.png"),fig,px_per_unit = 4)
            @debug "Saved bandstructure.pdf \n\"$plotpath\""
        else
            @warn "Could not save bandstructure plots"
        end
    catch e
        @warn "In plotbandstructure2d(sim::Simulation{T})"
        @error e
    end   
end


function plotbandstructure1d(sim::Simulation{T};plotkgrid=true) where {T<:Real}
    
    p       = getparams(sim)
    kxs     = p.kxsamples
end