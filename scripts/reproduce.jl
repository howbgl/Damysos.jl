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