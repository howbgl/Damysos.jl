export TwoBandDephasingLiouvillian

struct TwoBandDephasingLiouvillian{T<:Real} <: Liouvillian{T}
    hamiltonian::GeneralTwoBand{T}
    t1::T
    t2::T
end

function TwoBandDephasingLiouvillian(h::GeneralTwoBand,t1::Real,t2::Real)
    TwoBandDephasingLiouvillian(h,promote(t1,t2)...)
end

getparams(l::TwoBandDephasingLiouvillian) = (getparams(l.hamiltonian)...,t1=l.t1,t2=l.t2)
getparamsonly(l::TwoBandDephasingLiouvillian) = (t1=l.t1,t2=l.t2)

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