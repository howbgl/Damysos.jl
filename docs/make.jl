using Damysos
using Documenter

DocMeta.setdocmeta!(Damysos, :DocTestSetup, :(using Damysos); recursive=true)

makedocs(;
    modules=[Damysos],
    authors="Wolfgang Hogger <wolfgang.hogger@ur.de>",
    repo="https://git.uni-regensburg.de/how09898/Damysos.jl/blob/{commit}{path}#{line}",
    sitename="Damysos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://how09898.gitlab.io/Damysos.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
