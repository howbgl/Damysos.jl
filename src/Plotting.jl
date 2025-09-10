
export plotbandstructure
export plotdata
export plotfield


const DEFAULT_FIGSIZE           = (1200,800)
const DEFAULT_MAX_HARMONIC      = 40
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
    # Label(f[1,2],sidelabel,tellheight=false,justification = :left)
    
    return f
end

function calculate_spectrum(timeseries::Vector{<:Real},dt::Real;
    fftwindow=hanning)

    pdg = periodogram(
        timeseries,
        nfft=8*length(timeseries),
        fs=1/dt,
        window=fftwindow)
    ydata       = pdg.power .* (pdg.freq .^ 2)
    return pdg.freq,ydata
end

function plotpowerspectra(timeseries::Vector{Vector{T}},
                    labels::Vector{String},
                    frequencies::Vector{T},
                    timesteps::Vector{T};
                    maxharm=DEFAULT_MAX_HARMONIC,
                    fftwindow=hanning,
                    title="",
                    sidelabel="",
                    colors="categorical",
                    xlims=nothing,
                    ylims=nothing,
                    noiselevel_mark=nothing,
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

    total_ymax = typemin(T)

    for (i,data,label,dt,ν) in zip(1:length(timeseries),timeseries,labels,timesteps,
                                    frequencies)
        xdata,ydata = calculate_spectrum(data,dt,fftwindow=fftwindow)
        ymax        = maximum(ydata)
        total_ymax  = total_ymax < ymax ? ymax : total_ymax
        xdata       = 1/ν .* xdata
        cut_inds    = ydata .> 10floatmin(T)
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
    # Label(f[1,2],sidelabel,tellheight=false,justification = :left)

    if !isnothing(noiselevel_mark)
        hlines!(ax,[noiselevel_mark*total_ymax];
            color=:red,
            linestyle=:dash,
            label="$(noiselevel_mark) * maximum")
    end
    axislegend(ax,position=:lb)

    !isnothing(xlims) && xlims!(ax,xlims)
    !isnothing(ylims) && ylims!(ax,ylims)
    
    return f
end

function plotdata(sims::Vector{<:Simulation}, path::String = pwd();
    maxharm=DEFAULT_MAX_HARMONIC,
    fftwindow=hanning,
    kwargs...)
    
    for obs in sims[1].observables
        @info "Plotting " * getshortname(obs)
        try
            plotdata(
                sims,
                obs,
                path;
                maxharm=maxharm,
                fftwindow=fftwindow,
                kwargs...)
        catch e
            @warn "Plotting of $(getshortname(obs)) failed"
            showerror(stdout, e)
            continue
        end
    end
end


function plotdata(
        sims::Vector{<:Simulation}, 
        vel::Union{Velocity{T}, VelocityX{T}}, 
        path::String = pwd();
        maxharm=DEFAULT_MAX_HARMONIC,
        fftwindow=hanning,
        title = stringexpand_vector([s.id for s in sims]),
        noiselevel_mark=nothing,
        kwargs...) where {T<:Real}

        collection = if vel isa Velocity
            zip([:vx,:vxintra,:vxinter,:vy,:vyintra,:vyinter],
                            ["vx","vxintra","vxinter","vy","vyintra","vyinter"])
        else
            zip([:vx,:vxintra,:vxinter], ["vx","vxintra","vxinter"])
        end
        veltype = typeof(vel)

    for (vsymb,vname) in collection

        timeseries  = Vector{Vector{T}}(undef,0)
        tsamples    = Vector{Vector{T}}(undef,0)
        timesteps   = Vector{T}(undef,0)
        frequencies = Vector{T}(undef,0)
        labels      = Vector{String}(undef,0)

        for sim in sims
            lc_in_nm    = ustrip(u"nm",lengthscaleSI(sim.unitscaling))
            ν           = reference_frequency_plotting(sim.drivingfield)
            d           = sim.dimensions
            ts_in_cyc   = gettsamples(sim) .* ν
            v           = filter(x -> x isa veltype,sim.observables)[1]
            v_nm_per_s  = v -> ustrip(u"nm/s",velocitySI(v,sim.unitscaling))
            data        = v_nm_per_s.(getproperty(v,vsymb))

            push!(timeseries,data)
            push!(tsamples,ts_in_cyc)
            push!(timesteps,getdt(sim))
            push!(frequencies,ν)
            push!(labels,sim.id)
        end

        figtime     = plottimeseries(
            timeseries,
            labels,
            tsamples,
            title=vname * " (" * title * ")",
            colors="continuous",
            ylabel=sims[1].dimensions == 1 ? "v [vF nm^-1]" : "v [vF nm^-2]",
            xlabel="time [1/ν_ref]",
            kwargs...)
        figspectra  = plotpowerspectra(timeseries,
            labels,
            frequencies,
            timesteps,
            maxharm=maxharm,
            fftwindow=fftwindow,
            title=vname * " (" * title * ")",
            xlabel="time [ν/ν_ref]",
            colors="continuous",
            noiselevel_mark=noiselevel_mark,
            kwargs...)
        
        ensuredirpath(path)
        CairoMakie.save(joinpath(path,vname*".png"),figtime, px_per_unit = 4)
        CairoMakie.save(joinpath(path,vname*"_spec.png"), figspectra, px_per_unit = 4)
        @debug "Saved $(vname).png & $(vname)_spec.png at\n\"$path\""
    end
end


function plotdata(sims::Vector{<:Simulation}, ::Occupation{T}, path::String = pwd();
    title = stringexpand_vector([s.id for s in sims]),
    kwargs...) where {T<:Real}

    timeseries  = Vector{Vector{T}}(undef,0)
    tsamples    = Vector{Vector{T}}(undef,0)
    timesteps   = Vector{T}(undef,0)
    frequencies = Vector{T}(undef,0)
    labels      = Vector{String}(undef,0)

    for sim in sims

        lc      = sim.unitscaling.lengthscale
        ν       = reference_frequency_plotting(sim.drivingfield)
        o       = filter(x -> x isa Occupation,sim.observables)[1]
        data    = o.cbocc  / (lc^sim.dimensions)

        push!(timeseries,data)
        push!(tsamples,collect(gettsamples(sim)))
        push!(timesteps,getdt(sim))
        push!(frequencies,ν)
        push!(labels,sim.id)
    end

    figtime = plottimeseries(
        timeseries,
        labels,
        tsamples;
        title="CB occupation" * "(" * title * ")",
        xlabel="time [1/ν_ref]",
        colors="continuous",
        kwargs...)

    ensuredirpath(path)
    CairoMakie.save(joinpath(path,"cb_occ.png"),figtime,px_per_unit = 4)
    @debug "Saved cb_occ.png at \n\"$path\""
end

function plotdata(
    sim::Simulation,
    path = joinpath(pwd(),getname(sim));
    fftwindow=hanning,
    maxharm=DEFAULT_MAX_HARMONIC,
    kwargs...)
    
    @info "Generating plots"

    for obs in sim.observables
        @info "Plotting " * getshortname(obs)
        try
            plotdata(sim,obs,path;
                fftwindow=fftwindow,
                maxharm=maxharm,
                kwargs...)
        catch e
            @warn "Plotting of $(getshortname(obs)) failed due to $e"
            continue
        end
    end

    plotfield(sim, path)
    plotbandstructure(sim, path)
end


function plotdata(
    sim::Simulation,
    vel::VelocityX,
    path::String = joinpath(pwd(),getname(sim));
    fftwindow=hanning,
    maxharm=DEFAULT_MAX_HARMONIC,
    kwargs...)

    dt          = getdt(sim)
    ν           = reference_frequency_plotting(sim.drivingfield)
    ν_SI        = frequencySI(ν,sim.unitscaling)
    ts_in_cyc   = collect(gettsamples(sim)) .* ν
    velocities  = [vel.vx,vel.vxintra,vel.vxinter]
    v_nm_per_s  = v -> ustrip(u"nm/s",velocitySI(v,sim.unitscaling))
    timeseriesx = [v_nm_per_s.(x) for x in velocities]
    tsamplesx   = [ts_in_cyc,ts_in_cyc,ts_in_cyc]
    timestepsx  = [dt,dt,dt]
    frequenciesx = [ν,ν,ν]
    labelsx     = ["vx", "vxintra", "vxinter"]

    timeseries  = [timeseriesx]
    tsamples    = [tsamplesx]
    timesteps   = [timestepsx]
    frequencies = [frequenciesx]
    labels      = [labelsx]

    for (data,lab,ts,dt,ν) in zip(timeseries,labels,tsamples,timesteps,frequencies)

        figtime     = plottimeseries(
            data,
            lab,
            ts,
            title=sim.id,
            sidelabel=printparamsSI(sim),
            ylabel=sim.dimensions == 1 ? "v [vF nm^-1]" : "v [vF nm^-2]",
            xlabel="time [1/$(ν_SI)]",
            kwargs...)
        figspectra  = plotpowerspectra(
            data,
            lab,
            ν,
            dt,
            maxharm=maxharm,
            fftwindow=fftwindow,
            title=sim.id,
            sidelabel=printparamsSI(sim),
            kwargs...)

        ensuredirpath(path)
        CairoMakie.save(joinpath(path,"$(lab[1]).png"),figtime,px_per_unit = 4)
        CairoMakie.save(joinpath(path,"$(lab[1])_spec.png"),
            figspectra,px_per_unit = 4)
        @debug "Saved $(lab[1]).png & at $(lab[1])_spec.png \n\"$path\""
    end

    return nothing
end


function plotdata(
    sim::Simulation,
    vel::Velocity,
    path::String = joinpath(pwd(),getname(sim));
    fftwindow=hanning,
    maxharm=DEFAULT_MAX_HARMONIC,
    kwargs...)

    dt          = getdt(sim)
    ν           = reference_frequency_plotting(sim.drivingfield)
    ν_SI        = frequencySI(ν,sim.unitscaling)
    d           = sim.dimensions
    ts_in_cyc   = collect(gettsamples(sim)) .* ν
    velocities  = [vel.vx,vel.vxintra,vel.vxinter]
    v_nm_per_s  = v -> ustrip(u"nm/s",velocitySI(v,sim.unitscaling))
    timeseriesx = [v_nm_per_s.(x) for x in velocities]
    tsamplesx   = [ts_in_cyc,ts_in_cyc,ts_in_cyc]
    timestepsx  = [dt,dt,dt]
    frequenciesx = [ν,ν,ν]
    labelsx     = ["vx", "vxintra", "vxinter"]

    timeseries  = [timeseriesx]
    tsamples    = [tsamplesx]
    timesteps   = [timestepsx]
    frequencies = [frequenciesx]
    labels      = [labelsx]

    if sim.dimensions==2
        velocities  = [vel.vy,vel.vyintra,vel.vyinter]
        timeseriesy = [v_nm_per_s.(x) for x in velocities]
        push!(timeseries,timeseriesy)
        push!(tsamples,tsamplesx)
        push!(timesteps,timestepsx)
        push!(frequencies,frequenciesx)
        push!(labels,["vy","vyintra","vyinter"])
    end

    for (data,lab,ts,dt,ν) in zip(timeseries,labels,tsamples,timesteps,frequencies)

        figtime     = plottimeseries(
            data,
            lab,
            ts,
            title=sim.id,
            sidelabel=printparamsSI(sim),
            ylabel=sim.dimensions == 1 ? "v [vF nm^-1]" : "v [vF nm^-2]",
            xlabel="time [1/$(ν_SI)]",
            kwargs...)
        figspectra  = plotpowerspectra(
            data,
            lab,
            ν,
            dt,
            maxharm=maxharm,
            fftwindow=fftwindow,
            title=sim.id,
            sidelabel=printparamsSI(sim),
            kwargs...)

        ensuredirpath(path)
        CairoMakie.save(joinpath(path,"$(lab[1]).png"),figtime,px_per_unit = 4)
        CairoMakie.save(joinpath(path,"$(lab[1])_spec.png"),
            figspectra,px_per_unit = 4)
        @debug "Saved $(lab[1]).png & $(lab[1])_spec.png at \n\"$path\""
    end
        

    return nothing
end


function plotdata(
    sim::Simulation{T},
    occ::Occupation{T},
    path::String = joinpath(pwd(),getname(sim));
    kwargs...) where {T<:Real}

    dt          = getdt(sim)
    ν           = reference_frequency_plotting(sim.drivingfield)
    ν_SI        = frequencySI(ν,sim.unitscaling)
    lc_in_nm    = ustrip(u"nm",lengthscaleSI(sim.unitscaling))
    d           = sim.dimensions
    data        = occ.cbocc ./ lc_in_nm^d
    ts_in_cyc   = collect(gettsamples(sim)) .* ν

    figtime     = plottimeseries(
        [data],
        ["CB occupation"],
        [ts_in_cyc];
        title=sim.id,
        sidelabel=printparamsSI(sim),
        xlabel="time [1/$(ν_SI)]",
        ylabel = sim.dimensions == 1 ? "ρcc [nm^-1]" : "ρcc [nm^-2]",
        kwargs...)

   ensuredirpath(path)
    CairoMakie.save(joinpath(path,"cb_occ.png"),figtime,px_per_unit = 4)
    @debug "Saved cb_occ.png at \"$path\"\n"
    
end

function plotdata(
    s::Union{Simulation,Vector{<:Simulation}},
    obs::Observable, 
    path::String = joinpath(pwd(),getname(sim));
    kwargs...)
    
    @warn "Plotting of $(getshortname(obs)) not implemented"

    return nothing
end

function plotfield(sim::Simulation,path::String = joinpath(pwd(),getname(sim)))

    @info "Plotting driving field"

    name    = getname(sim)
    ts      = collect(gettsamples(sim))
    ax      = get_vecpotx(sim)
    ay      = get_vecpoty(sim)
    ex      = get_efieldx(sim)
    ey      = get_efieldy(sim)

    ν                   = reference_frequency_plotting(sim.drivingfield)
    ν_SI                = frequencySI(ν,sim.unitscaling)
    ts_in_cyc           = ts .* ν
    tc                  = timescaleSI(sim.unitscaling)
    lc                  = lengthscaleSI(sim.unitscaling)
    vecpot_SI_factor    = ustrip(u"MV*fs/cm",Unitful.ħ/(q_e*lc))
    field_SI_factor     = ustrip(u"MV/cm",Unitful.ħ/(q_e*lc*tc))
    try
        figa    = plottimeseries(
            [ax.(ts) .* vecpot_SI_factor,ay.(ts) .* vecpot_SI_factor],
            ["Ax","Ay"],
            [ts_in_cyc,ts_in_cyc],
            title=name,
            sidelabel=printparamsSI(sim),
            xlabel="time [1/$(ν_SI)]",
            ylabel="vector potential [fs MV/cm]")
        fige    = plottimeseries(
            [ex.(ts) .* field_SI_factor,ey.(ts) .* field_SI_factor],
            ["Ex","Ey"],
            [ts_in_cyc,ts_in_cyc],
            title=name,
            sidelabel=printparamsSI(sim),
            xlabel="time [1/$(ν_SI)]",
            ylabel="el. field [MV/cm]")

        (success,path)  = ensuredirpath([path])
        if success
            CairoMakie.save(joinpath(path,"vecfield.png"),figa,px_per_unit = 4)
            CairoMakie.save(joinpath(path,"efield.png"),fige,px_per_unit = 4)
            @debug "Saved vecfield.png & efield.spec.png at \"$path\"\n"
        else
            @warn "Could not save driving field plots"
        end
    catch e
        @warn "In plotfield(sim::Simulation{T})"
        @error e
    end
end

function plotbandstructure(sim::Simulation,path::String = joinpath(pwd(),getname(sim));
    plotkgrid=false)
    
    if sim.dimensions==2
        return plotbandstructure2d(sim,path;plotkgrid=plotkgrid)
    elseif sim.dimensions==1
        return plotbandstructure1d(sim,path;plotkgrid=plotkgrid)
    end
end


function plotbandstructure2d(sim::Simulation,path::String = joinpath(pwd(),getname(sim));
    plotkgrid=false,nk=2048)
    
    @info "Plotting bandstructure"
    
    bz          = getbzbounds(sim)
    bzSI        = [ustrip(u"Å^-1",wavenumberSI(k,sim.unitscaling)) for k in bz]
    bzSI_kx     = [bzSI[1],bzSI[2],bzSI[2],bzSI[1],bzSI[1]] 
    bzSI_ky     = [bzSI[3],bzSI[3],bzSI[4],bzSI[4],bzSI[3]] 
    Δϵ          = getΔϵ(sim)
    kmax        = maximum([maximum(getkxsamples(sim)),maximum(getkysamples(sim))])
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
            for ky in getkysamples(sim)
                scatter!(ax,collect(getkxsamples(sim)),fill(ky,getnkx(sim));
                    color=:black,
                    markersize=1.2)
            end
        end
        lines!(ax,bzSI_kx,bzSI_ky,color=:black)
        tooltip!(bzSI[2],bzSI[4],"Brillouin Zone",offset=0,align=0.8)
        # Label(fig[1,3],printparamsSI(sim),tellheight=false,justification = :left)

        (success,path)  = ensuredirpath([path])
        if success
            CairoMakie.save(joinpath(path,"bandstructure.png"),fig,px_per_unit = 4)
            @debug "Saved bandstructure.png \n\"$path\""
        else
            @warn "Could not save bandstructure plots"
        end
    catch e
        @warn "In plotbandstructure2d(sim::Simulation{T})"
        @error e
    end   
end


function plotbandstructure1d(sim::Simulation{T},path::String = joinpath(pwd(),getname(sim));
    plotkgrid=true) where {T<:Real}
    
    # TODO implement plotbandstructure1d
end