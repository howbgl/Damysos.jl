# Use the official Julia image as the base
FROM julia:latest

# Create and set the working directory
WORKDIR /app

# Copy your package code to the container
COPY . /app

# Install dependencies
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.add("PackageCompiler")'

# Create a custom system image with your package precompiled
RUN julia --project=. -e '
    using Pkg;
    using PackageCompiler;
    Pkg.precompile();
    create_sysimage(["Damysos", "Documenter"], sysimage_path="sysimage.so", precompile_execution_file="docs/make.jl")
'

# Make sure the sysimage is in the right place
RUN mv sysimage.so /app/sysimage.so
