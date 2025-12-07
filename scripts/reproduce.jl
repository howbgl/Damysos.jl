##########################################################
# Example script for reproducing published results
##########################################################
#
# * published in
# * data: https://doi.org/10.5281/zenodo.8341513 

# Example data file included in repository at rawdata/Fig2_data.hdf5
# To reproduce other results, download the corresponding data file
# from the Zenodo link above and change the path below accordingly.

using Damysos,HDF5

# Load the the simulation data (includes metadata needed to reconstruct the simulation)
simulation = h5open("rawdata/Fig2_data.hdf5", "r") do f
    load_obj_hdf5(f["zeta=7.5_M=0.177"])
end

# Choose solver:
# - LinearChunked(kchunksize::Integer, ...) for CPU
# - LinearCUDA(kchunksize::Integer, ...) for GPU
# Choose chunk size according to available memory

const solver = LinearChunked(1_024)
const fns = define_functions(simulation,solver)

# Run the simulation
const results = run!(simulation,fns,solver; savepath="scripts/reproduced_data")