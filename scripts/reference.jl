using Unitful,LoggingExtras,Dates,Formatting,DifferentialEquations,TerminalLoggers
using Distributed,BenchmarkTools

@everywhere using Damysos
@everywhere using StaticArrays


import Damysos.getshortname
import Damysos.ensurepath
import Damysos.buildrhs_cc_cv_x_expression
import Damysos.buildobservable_expression_upt
import Damysos.buildbzmask_expression_upt


function make_teelogger(logging_path::AbstractString,name::AbstractString)

    ensurepath(logging_path)
    info_filelogger  = FileLogger(joinpath(logging_path,name)*"_$(now()).log")
    info_logger      = MinLevelLogger(info_filelogger,Logging.Info)
    all_filelogger   = FileLogger(joinpath(logging_path,name)*"_$(now())_debug.log")

    return  TeeLogger(TerminalLogger(),info_logger,all_filelogger)
end

global_logger(TerminalLogger())

@everywhere function make_system()

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
    l       = TwoBandDephasingLiouvillian(h,Inf,timescaled(t2,us))
    df      = GaussianAPulse(us,σ,freq,emax)
    pars    = NumericalParams2d(dkx,dky,kxmax,kymax,dt,-5df.σ)
    obs     = [Velocity(h)]

#  id      = sprintf1("%x",hash([h,df,pars,obs,us]))
    id      = "ref"
    name    = "Simulation{$(typeof(h.m))}(2d)reference"
    dpath   = "/home/how09898/phd/data/hhgjl/expressions_test/reference"
    ppath   = "/home/how09898/phd/plots/hhgjl/expressions_test/reference"

    return Simulation(l,df,pars,obs,us,2,id,dpath,ppath)
end


const ppath   = "/home/how09898/phd/plots/hhgjl/expressions_test/reference"
const id      = "ref"
global_logger(make_teelogger(ppath,id))
@info "Now saving logs to $ppath"
@everywhere sim     = make_system()

@everywhere const functions = define_functions(sim)
@info "Solving differential equations"
const observables,time,rest... = @timed run!(sim,functions,CPULinearChunked(512))
@info "Call to run! took $(time/60.)min"



