using Pkg
using PackageCompiler

# Activate your project
Pkg.activate(".")

# Precompile dependencies to speed up sysimage creation
Pkg.precompile()

# Create a custom system image
create_sysimage(
    ["Damysos", "Documenter"],  # Add other dependencies if needed
    sysimage_path="sysimage.so",
    precompile_execution_file="docs/make.jl"  # Optional: Helps include relevant precompilation
)
