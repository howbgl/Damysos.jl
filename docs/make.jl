using Documenter
using Damysos

DocMeta.setdocmeta!(Damysos, :DocTestSetup, :(using Damysos); recursive=true)

makedocs(;
    modules=[Damysos],
    authors="Wolfgang Hogger <wolfgang.hogger@gmail.com>",
    repo="https://howbgl.github.io/Damysos.jl/",
    sitename="Damysos.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        # canonical="https://how09898.gitlab.io/Damysos.jl",
        edit_link="main",
        assets=String[],
    ),
    deploydocs = [
        Documenter.GitHubActions(),
        push_preview = true,  # Enable deployment for unregistered packages
    ],
    pages=[
        "Home" => "index.md",
        "Two-band Hamiltonians" => "twoband.md",
        "reference.md",
    ],
)
