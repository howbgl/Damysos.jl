using Documenter
using Damysos

import Documenter.Remotes.GitHub

DocMeta.setdocmeta!(Damysos, :DocTestSetup, :(using Damysos); recursive=true)

makedocs(;
    modules=[Damysos],
    authors="Wolfgang Hogger <wolfgang.hogger@gmail.com>",
    repo=GitHub("howbgl", "Damysos.jl"),
    sitename="Damysos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        # canonical="https://how09898.gitlab.io/Damysos.jl",
        edit_link="main",
    ),
    pages=[
        "Home" => "index.md",
        "Two-band Hamiltonians" => "twoband.md",
        "reference.md",
    ],
)

deploydocs(
    repo="https://github.com/howbgl/Damysos.jl.git",
    devbranch="main",
    versions = [
        "dev" => "dev",
        "v1.0.0" => "v1.0.0",
        "stable" => "v^",
    ],
)