using Damysos,TerminalLoggers,Dates,LoggingExtras

function make_teelogger(logging_path::AbstractString,name::AbstractString)

    ensurepath(logging_path)
    info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
    info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
    all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

    return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end


function make_demo_simulation(datapath="~/data",plotpath=datapath)

    vf        = u"4.3e5m/s"
    freq      = u"5THz"
    m         = u"20.0meV"
    emax      = u"0.1MV/cm"
    tcycle    = uconvert(u"fs",1/freq) # 100 fs
    t2        = tcycle / 4             # 25 fs
    t1        = Inf*u"1s"
    σ         = u"800.0fs"

    # converged at
    # dt = 0.01
    # dkx = 1.0
    # dky = 1.0
    # kxmax = 175
    # kymax = 100

    dt      = 0.01
    dkx     = 1.0
    kxmax   = 175.0
    dky     = 1.0
    kymax   = 100.0

    us      = scaledriving_frequency(freq,vf)
    h       = GappedDirac(energyscaled(m,us))
    l       = TwoBandDephasingLiouvillian(h,timescaled(t1,us),timescaled(t2,us))
    df      = GaussianAPulse(us,σ,freq,emax)
    pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
    obs     = [Velocity(pars),Occupation(pars)]

    id      = "demo"

    return Simulation(l,df,pars,obs,us,id,datapath,plotpath)
end

const sim = make_demo_simulation()
const logger = make_teelogger(sim.plotpath,sim.id)
const solver = LinearChunked(1_024)
const fns = define_functions(sim,solver)

global_logger(logger)
@info "$(now())\nOn $(gethostname()):"

const results,time,rest... = @timed run!(sim,fns,solver)

@info "$(time/60.)min spent in run!(...)"
@debug rest
@info "$(now()): calculation finished."