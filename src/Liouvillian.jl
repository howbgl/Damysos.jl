export TwoBandDephasingLiouvillian
"""
    TwoBandDephasingLiouvillian{T<:Real} <: Liouvillian{T}

Represents a system with a two-band Hamiltonian and ``T_2`` dephasing.

# See also
[`GappedDirac`](@ref GappedDirac), [`GeneralTwoBand`](@ref GeneralTwoBand)
"""
struct TwoBandDephasingLiouvillian{T<:Real} <: Liouvillian{T}
    hamiltonian::GeneralTwoBand{T}
    t1::T
    t2::T
end

function TwoBandDephasingLiouvillian(h::GeneralTwoBand,t1::Real,t2::Real)
    TwoBandDephasingLiouvillian(h,promote(t1,t2)...)
end

for func = (BAND_SYMBOLS...,DIPOLE_SYMBOLS...,VELOCITY_SYMBOLS...)
    @eval(Damysos,$func(l::Liouvillian) = $func(l.hamiltonian))
end

getparams(l::TwoBandDephasingLiouvillian) = (getparams(l.hamiltonian)...,t1=l.t1,t2=l.t2)
getparamsonly(l::TwoBandDephasingLiouvillian) = (t1=l.t1,t2=l.t2)
function getshortname(l::TwoBandDephasingLiouvillian)  
    return "TwoBandDephasingLiouvillian($(getshortname(l.hamiltonian)))"
end

function getparamsSI(l::TwoBandDephasingLiouvillian,us::UnitScaling)
    t1 = timeSI(l.t1,us)
    t2 = timeSI(l.t2,us)
    return (t1=t1,t2=t2)
end

function Base.show(io::IO,::MIME"text/plain",l::Liouvillian) 
    println(io,getshortname(l))
    println(io,"  Hamiltonian: $(getshortname(l.hamiltonian))")
    print(io,prepend_spaces(stringexpand_nt(getparams(l.hamiltonian)),2))
    print(io,"\n")
    print(io,l |> getparamsonly |> stringexpand_nt |> prepend_spaces)
end

function printparamsSI(l::TwoBandDephasingLiouvillian,us::UnitScaling;digits=3)

    symbols     = [:t1,:t2]
    valuesSI    = [timeSI(l.t1,us),timeSI(l.t2,us)]
    values      = [l.t1,l.t2]

    str = ""

    for (s,v,vsi) in zip([:t1,:t2],values,valuesSI)
        valSI   = round(typeof(vsi),vsi,sigdigits=digits)
        val     = round(v,sigdigits=digits)
        str     *= "$s = $valSI ($val)\n"
    end
    return printparamsSI(l.hamiltonian,us,digits=digits) * str
end
