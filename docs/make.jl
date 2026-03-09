using Documenter
using Damysos
import Documenter.Remotes.GitHub

DocMeta.setdocmeta!(Damysos, :DocTestSetup, :(using Damysos); recursive=true)

makedocs(;
    modules=[Damysos],
    authors="Wolfgang Hogger ",
    repo=GitHub("howbgl", "Damysos.jl"),
    sitename="Damysos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Two-band Hamiltonians" => "twoband.md",
        "reference.md",
    ],
)

deploydocs(
    repo="github.com/howbgl/Damysos.jl.git",
    devbranch="main",
    versions = [
        "stable" => "v^",
        "v#.#.#",
        "dev" => "dev",
    ],
)